import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import block_diag
from scipy.optimize import linprog
from scipy.spatial import distance_matrix

import random
import pandas as pd
import numpy as np
import json
import geopandas as gpd
from brix import Handler, Indicator
import pandana
import datetime
import requests

def get_crs(gdf):
    avg_lng=gdf.unary_union.centroid.x
    utm_zone = int(np.floor((avg_lng + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    return utm_crs

def find_closest_nodes(nodes_gdf, places):
    nodes_crs=get_crs(nodes_gdf)
    nodes_gdf_projected=nodes_gdf.to_crs(nodes_crs)  
    nodes_kdtree=KDTree([[p.x, p.y] for p in nodes_gdf_projected.geometry])
    places_proj=places.to_crs(nodes_crs)
    dist, ind_nearest=nodes_kdtree.query([[c.x, c.y] for c in places_proj.geometry.centroid])
    nearest_nodes=[nodes_gdf_projected.index[i] for i in ind_nearest]
    return dist, nearest_nodes

def get_reachable_matrix(nodes_gdf, all_places, pdna_net, max_dist, return_dist_mat=False):
    dist_to_nearest_nodes, nearest_nodes=find_closest_nodes(nodes_gdf, all_places)
    all_places['dist_to_nearest_nodes']=dist_to_nearest_nodes
    all_places['nearest_nodes']=nearest_nodes
    dist_mat=[]
    for ind, row in all_places.iterrows():
        origin_nodes=[row['nearest_nodes']]*len(all_places)
        dest_nodes=nearest_nodes
        path_lengths=pdna_net.shortest_path_lengths(origin_nodes, dest_nodes)
        path_lengths_with_ends=np.array(path_lengths)+1.4*np.array(dist_to_nearest_nodes)+1.4*row['dist_to_nearest_nodes']
        dist_mat.append(list(path_lengths_with_ends))
    dist_mat=np.array(dist_mat)
    reachable=dist_mat<max_dist
    if return_dist_mat:
        return reachable, dist_mat
    else:
        return reachable

def calculate_access(all_places, target_settings, reachable):
    # all target columns should already be in terms of population capacity
    # i.e. restaurant column indicates the population that can be served by each restaurant
    
    # simple access- not considering demand or compeition
    target_names=list(target_settings.keys())
    target_columns=[target_settings[t]['column'] for t in target_names]

    access=np.dot(reachable, all_places[target_columns].fillna(0))

    # get demand for each target in each place
    for t in target_names:
        all_places['demand_{}'.format(t)]=all_places[target_settings[t]['demand_source']].sum(axis=1)

    # get the demand that is competing for each locations' capacity
    feed_in=np.dot(reachable.T, all_places[['demand_{}'.format(t) for t in target_names]].fillna(0))

    # get the normalised capacity of every target
    all_places[['{}_norm'.format(t) for t in target_names]]=(all_places[target_columns]/feed_in).replace(
        [np.inf, -np.inf], 0).fillna(0)
    
    # get the normalised access at every place
    norm_access=np.dot(reachable, 
                       all_places[['{}_norm'.format(t) for t in target_names]])
    norm_access=np.clip(norm_access, 0, 1)

    scores={}
    for it, t in enumerate(target_names):
        col_name='{}_access'.format(t)
        demand=all_places['demand_{}'.format(t)]
        ind_source=((demand>0)
                    &(all_places['impact_area'])
                   )
        all_places.loc[ind_source, col_name]=norm_access[ind_source, it]

        scores[t]=(all_places.loc[ind_source, col_name]*demand.loc[ind_source]).sum()/(demand.loc[ind_source]).sum()
        # print('{}: {}'.format(t, scores[t]))
    geo_output=all_places.loc[((all_places['impact_area']==True)|(all_places['impact_area']==True)),
        ['{}_access'.format(t) for t in target_names]+['impact_area','geometry']]
    return geo_output, scores


class Proximizer():
    """ Model
    Everything is in units of population-carrying-capacity. 
    eg. a city needs 2 restaurants per 1000 people -> each restaurant has capacity of 500

    maximise met demand for each target such that:
    - total met demand is the sum of met Demand over each target X (obj)
    - met demand for X is the sum over each source of the product of demand_X and normalised_accessibility_X (eq 1)
    - normalised_accessibility_X is the lesser of 1 (ineq 2) and the sum of normalised capacities of X in all reachable places (ineq 3)
    - normalised capacity of X in a place is capacity_p^x divided by inverse-access_p^x (eq 2)
    - different from previous model:
        - capacity_p^x is persons_per_POI^x * N_units_p^x (where only N_units is a variable) (fixed input: potential_target_capacities)
        - for each place, the sum over targets of units_p^x * Area^x can't exceed the area of the place (A_p: fixed input) (ineq 1)
        - met demand for X only sums over impact area (other areas still included as they still compete for capacity)

    Design vector:
    - Total_met_demand: [T^x1, T^x2 ...] (one entry each target type)
    - Assigned units: [N_p1^x1, N_p2^x1 .... N_p1^x2, N_p2^x2....] (one entry each target type and design space)
    - Normalised Capacity: [nC_p1^x1, nC_p2^x1 ...  ...] (one entry each target type and design space)
    - Normalised Accessibility: [nA_s1^x1, nA_s2^x1 ...  ...] (one entry each target type and source space)

    """
    def __init__(self, reachable_s_t, demand_mat, ind_source_in_impact_area,
                 potential_target_capacities, targets, 
                 sqm_spaces, sqm_p_poi, norm_access_static=None, integer_amenities=True):
        self.n_source, self.n_design = reachable_s_t.shape
        self.n_targets=demand_mat.shape[1]
        invA = np.dot(reachable_s_t.T,demand_mat)
        inv_invA=1/invA
        inv_invA[inv_invA == np.inf] = 0
        self.A_ub, self.b_ub = self.create_ineq_constraints(
            reachable_s_t, norm_access_static, sqm_spaces, sqm_p_poi)
        self.A_eq, self.b_eq = self.create_eq_constraints(demand_mat, inv_invA, potential_target_capacities, ind_source_in_impact_area)
        self.c = self.create_obj_function(targets)
        self.integer_amenities=integer_amenities
        
    def create_eq_constraints(self, demand_mat, inv_invA, potential_target_capacities, ind_source_in_impact_area):
        n_targets, n_design, n_source = self.n_targets, self.n_design, self.n_source
        # EQ CONSTRAINT 1: met demand of x is the sum over each source of the product of demand_X and normalised_accessibility_X
        demand_mat[~ind_source_in_impact_area, :]=0
        temp=block_diag(*[demand_mat[:,i] for i in range(n_targets)])

        Aeq1=np.column_stack([
            -np.identity(n_targets), # LHS: the totals
            np.zeros([n_targets, (2*n_targets)*n_design]), # ignore  target bools, norm capacities of design cells
            temp]) # RHS: sum of demand*norm_access over all sources
        beq1=np.zeros(n_targets)    
        
        # EQ CONSTRAINT 2: normalised target capacity is the capacity over the inverse access
        diag=np.multiply(inv_invA.flatten('F'), potential_target_capacities)

        Aeq2=np.column_stack(
            [np.zeros([n_targets*n_design, n_targets]), #ignore totals
             np.diag(diag), # RHS: bool * (potential capacity normalised by invAccess)
             -np.identity(n_targets*n_design), # LHS norm target capacity
             np.zeros([n_targets*n_design, n_targets*n_source])]) # ignore the norm access
        beq2=np.zeros(n_targets*n_design)

        A_eq=np.concatenate((Aeq1, Aeq2))
        b_eq=np.concatenate((beq1, beq2))
        return A_eq, b_eq
        
    def create_ineq_constraints(self, reachable_s_t, norm_access_static, sqm_spaces, sqm_p_poi):
        n_targets, n_design, n_source = self.n_targets, self.n_design, self.n_source
        # INEQUALITY CONSTRAINT 1: areas of assigned amenities can't exceed available area
        A1=np.column_stack([
            np.zeros([n_design, n_targets]), # total access vars
            np.column_stack([sqm*np.identity(n_design) for sqm in sqm_p_poi]),
            np.zeros([n_design, n_design*n_targets]), # norm capacity vars
            np.zeros([n_design, n_source*n_targets])]) # norm access vars

        b1 = sqm_spaces 

        # INEQUALITY CONSTRAINT 2: normalised access cant exceed 1
        A2=np.column_stack([
            np.zeros([n_source*n_targets, n_targets+n_design*(2*n_targets)]),# TODO: use separate lines for clarity
            np.identity(n_source*n_targets)])
        b2=np.ones(n_source*n_targets)

        block_diag_reachable=block_diag(*[reachable_s_t for i in range(n_targets)]).astype(int)

        # INEQUALITY CONSTRAINT 3: normalised access cant exceed the sum of the reachable normalised capacities
        A3 = np.column_stack([
            np.zeros([n_targets*n_source, n_targets]), #ignore totals
            np.zeros([n_targets*n_source, n_targets*n_design]), # ignore bools
            -block_diag_reachable, #RHS: sum capacity each reachable target
            np.identity(n_targets*n_source) # LHS: norm access each source, each type
        ])
        if norm_access_static is None:
            b3=np.zeros(n_targets*n_source)
        else:
            b3=norm_access_static.flatten('F')

        A_ub=np.concatenate((A1, A2, A3))
        b_ub=np.concatenate((b1, b2, b3))
        
        return A_ub, b_ub
        
    def create_obj_function(self, targets):
        n_targets, n_design, n_source = self.n_targets, self.n_design, self.n_source
        c = [-1 for t in targets] +[1e-6]*(n_design*n_targets) + [0]*(n_design*n_targets) + [0]*(n_source*n_targets)
        return c
        
    def solve(self):
        n_targets, n_design, n_source = self.n_targets, self.n_design, self.n_source
        method=None
        bounds=bounds=[(0, None) for i in range(len(self.c))]
        x0=None
        options=None

        integrality=([0]*n_targets + # total access is continuous
                     [int(self.integer_amenities)]*(n_design*n_targets) +  # amenity type dummys are int
                     [0]*(n_design*n_targets) +  # norm capacities are continuous
                     [0]*(n_source*n_targets)    # norm accessibilities are continuous
                    )

        result=linprog(c=self.c, 
                       A_ub=self.A_ub, b_ub=self.b_ub, 
                       A_eq=self.A_eq, b_eq=self.b_eq, 
                       bounds=bounds, 
                       x0=x0,
                       options=options, 
                       integrality=integrality
                      ) 
        print('Sucess: {}'.format(result.success))
        return result   
    
    def split_design_vector(self, x):
        n_targets, n_design, n_source = self.n_targets, self.n_design, self.n_source
        total_access=x[:n_targets]
        target_dummys=x[n_targets:(n_targets+(n_design*n_targets))]
        norm_capacities=x[(n_targets+(n_design*n_targets)):(n_targets+(n_design*(2*n_targets)))]
        norm_access = x[n_targets+(n_design*(2*n_targets)):]
        return total_access, target_dummys, norm_capacities, norm_access
    
    def split_slack(self, slack):
        n_targets, n_design, n_source = self.n_targets, self.n_design, self.n_source
        unmet_demand=slack[n_design:n_design+(n_source*n_targets)]
        over_supply=slack[-n_targets*n_source:]
        return unmet_demand, over_supply
        
    def split_by_target(self, targets, vector, n_per_target):
        n_targets=self.n_targets
        return {t: vector[n_per_target*i:n_per_target*(i+1)] for i, t in enumerate(list(targets))}

def expand_geogrid_data(geogrid_data):
    geogrid_data_gdf=geogrid_data.as_df()
    
    cell_area=(geogrid_data.get_geogrid_props()['header']['cellSize'])**2
    geogrid_data_gdf['height']=[h[1] if isinstance(h, list) else h for h in geogrid_data_gdf['height']]       
    geogrid_data_gdf['area']=cell_area
    is_interactive=geogrid_data_gdf['interactive'].astype(bool) # works for 'Web'
    geogrid_data_gdf['impact_area']=True
    geogrid_data_gdf['impact_area']=False
    geogrid_data_gdf.loc[geogrid_data_gdf['interactive'], 'impact_area']=True
    
    type_def=geogrid_data.get_type_info()
    
    # Add columns to each row (grid cell) for the sqm devoted to each :
    # CS_type, NAICS, LBCS and amenity
    present_types=[n for n in geogrid_data_gdf.loc[is_interactive, 'name'].unique() if 
                   ((n is not None) and (n!='None'))]
    for type_name in present_types:
        ind_this_type=((is_interactive)&(geogrid_data_gdf['name']==type_name))
        geogrid_data_gdf.loc[ind_this_type, '{}_area'.format(type_name)]=cell_area*geogrid_data_gdf.loc[ind_this_type,'height']
        for attr in ['sqmpp_res', 'sqmpp_emp']:
            if attr in type_def[type_name]:
                geogrid_data_gdf.loc[ind_this_type, attr]=type_def[type_name][attr]
        for attr in ['NAICS', 'LBCS', 'amenities']:
            if attr in type_def[type_name]:
                if type_def[type_name][attr] is not None:
                    for code in type_def[type_name][attr]:
                        col_name='sqm_{}_{}'.format(attr.lower(), code)
                        if col_name not in geogrid_data_gdf.columns:
                            geogrid_data_gdf[col_name]=0
                        code_prop=type_def[type_name][attr][code]
                        geogrid_data_gdf[col_name]+=(geogrid_data_gdf['{}_area'.format(type_name)]).fillna(0)*code_prop

    res_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_lbcs_1')]
    emp_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_naics_')]
    res_cols, emp_cols=[], []
    amenity_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_amenities_')]
    for col in res_sqm_cols:
        res_col_name=col.split('sqm_')[1]
        geogrid_data_gdf[res_col_name]=geogrid_data_gdf[col]/geogrid_data_gdf['sqmpp_res']
        res_cols.append(res_col_name)
    for col in emp_sqm_cols:
        emp_col_name=col.split('sqm_')[1]
        geogrid_data_gdf[emp_col_name]=geogrid_data_gdf[col]/geogrid_data_gdf['sqmpp_emp']
        emp_cols.append(emp_col_name)

    amenity_sqm_cols=[col for col in geogrid_data_gdf.columns if col.startswith('sqm_amenities_')]
    geogrid_data_gdf['amenity_total_sqm'] =geogrid_data_gdf[amenity_sqm_cols].sum(axis=1)    

    geogrid_data_gdf['res_total']=geogrid_data_gdf[res_cols].sum(axis=1)
    geogrid_data_gdf['emp_total']=geogrid_data_gdf[emp_cols].sum(axis=1)
    print('{} new residents'.format(geogrid_data_gdf['res_total'].sum()))
    print('{} new employees'.format(geogrid_data_gdf['emp_total'].sum()))
    return geogrid_data_gdf

class Proximity_Indicator(Indicator):
    def setup(self, static_places, geogrid, max_dist, 
              target_settings, pdna_net, nodes_gdf,sqm_pp_major,
              employed_ratio=1.7, reachable=None):
        print('Setting up Proximizer indicator')
        self.requires_geometry = True
        self.target_settings = target_settings
        self.indicator_type = 'hybrid'
        self.static_places=static_places
        self.sqm_pp_major=sqm_pp_major
        self.employed_ratio=employed_ratio
        self.target_list=[t for t in target_settings] # all target places including some whose locations are specified as inputs
        places=pd.concat([static_places, geogrid.loc[geogrid['interactive'].astype(bool)]], axis=0)
        if reachable is None:
            print('\t Getting reachable matrix')
            reachable=get_reachable_matrix(
                nodes_gdf, places, pdna_net, max_dist, return_dist_mat=False)
        self.reachable=reachable
        
        print('\t Calculating baseline scores')
        _ , scores=calculate_access(
            places, self.target_settings,  self.reachable)
        self.baseline_scores=scores
        
    def get_cs_heatmap(self, heatmap, heatmap_targets):
        cs_heatmap=heatmap.copy()
        # cs_heatmap.geometry=cs_heatmap.geometry.centroid
        cs_heatmap=cs_heatmap.__geo_interface__
        cs_heatmap['properties']=heatmap_targets
        for i_f, feat in enumerate(cs_heatmap['features']):
            prox_list=[feat['properties'][t] for t in heatmap_targets]
            cs_heatmap['features'][i_f]['properties']=prox_list
        return cs_heatmap
    
    def return_indicator(self, geogrid_data):
        start_time=datetime.datetime.now()
        geogrid_data_gdf=expand_geogrid_data(geogrid_data)
        
        is_interactive=geogrid_data_gdf['interactive'].astype(bool)
        #TODO: parks in person capacity units
        for a in self.sqm_pp_major:
            area_col=self.target_settings[a]['column']+'_area'
            if area_col in geogrid_data_gdf.columns:
                geogrid_data_gdf.loc[is_interactive, self.target_settings[a]['column']
                    ]=geogrid_data_gdf.loc[is_interactive, area_col]/self.sqm_pp_major[a]
        geogrid_data_gdf.loc[is_interactive, 'emp_capacity'
            ]=self.employed_ratio*geogrid_data_gdf.loc[is_interactive, 'emp_total']

        # Concatenate the baseline static data with the updated geogrid data
        updated_places=pd.concat([self.static_places, geogrid_data_gdf.loc[is_interactive]]).fillna(0)

        # Get the proximity scores and heatmap (before allocation of amenities)
        geo_output, pre_scores=calculate_access(
            updated_places, self.target_settings, self.reachable)
        final_scores=pre_scores

        heatmap_cols=[t+'_access' for t in self.target_list]+['geometry']
        geo_heatmap=geo_output.loc[geo_output['impact_area'], heatmap_cols]
        geo_heatmap.columns=self.target_list+['geometry']
        geo_heatmap.geometry=geo_heatmap.geometry.centroid
        cs_heatmap=self.get_cs_heatmap(geo_heatmap, self.target_list)
        result=[{'name': '{} Prox'.format(t).title(),
                 'description': 'Proximity of {} from {}'.format(t, self.target_settings[t]['demand_source']),
                 'value': final_scores[t],
                 'ref_value': self.baseline_scores[t]} for t in self.target_list]
        
        print('Time taken: {}'.format(datetime.datetime.now()-start_time))  
        return {'heatmap':cs_heatmap,'numeric':result}


class Proximize_Indicator(Indicator):
    def setup(self, static_places, geogrid, max_dist, amenities_generative,
              target_settings, pdna_net, nodes_gdf, 
              sqm_p_poi_generative, p_per_poi_generative,sqm_pp_major,
             reachable=None, integer_amenities=True):
        print('Setting up Proximizer indicator')
        self.requires_geometry = True
        self.target_settings = target_settings
        self.indicator_type = 'hybrid'
        self.static_places=static_places
        self.sqm_p_poi_generative=sqm_p_poi_generative
        self.p_per_poi_generative=p_per_poi_generative
        self.sqm_pp_major=sqm_pp_major
        self.integer_amenities=integer_amenities
        self.amenities_generative=amenities_generative # places to be allocated by algorithm
        self.target_list=[t for t in target_settings] # all target places including some whose locations are specified as inputs
        places=pd.concat([static_places, geogrid.loc[geogrid['interactive'].astype(bool)]], axis=0)
        if reachable is None:
            print('\t Getting reachable matrix')
            reachable=get_reachable_matrix(
                nodes_gdf, places, pdna_net, max_dist, return_dist_mat=False)
        self.reachable=reachable
        
        print('\t Calculating baseline scores')
        _ , scores=calculate_access(places, self.target_settings,  self.reachable)
        # scores['LW_Symmetry']=min(scores['Housing'], scores['Jobs'])
        # scores['Amenities']=sum([scores[a] for a in self.amenities_generative])/len(self.amenities_generative)
        self.baseline_scores=scores
        
    def get_cs_heatmap(self, heatmap, heatmap_targets):
        cs_heatmap=heatmap.copy()
        # cs_heatmap.geometry=cs_heatmap.geometry.centroid
        cs_heatmap=cs_heatmap.__geo_interface__
        cs_heatmap['properties']=heatmap_targets
        for i_f, feat in enumerate(cs_heatmap['features']):
            prox_list=[feat['properties'][t] for t in heatmap_targets]
            cs_heatmap['features'][i_f]['properties']=prox_list
        return cs_heatmap
    
    def return_indicator(self, geogrid_data):
        start_time=datetime.datetime.now()
        geogrid_data_gdf=expand_geogrid_data(geogrid_data)
        
        is_interactive=geogrid_data_gdf['interactive'].astype(bool)
        #TODO: parks in person capacity units
        for a in self.sqm_pp_major:
            area_col=a+'_area'
            if area_col in geogrid_data_gdf.columns:
                geogrid_data_gdf.loc[is_interactive, a+'_capacity'
                    ]=geogrid_data_gdf.loc[is_interactive, a+'_area']/self.sqm_pp_major[a]

        # Concatenate the baseline static data with the updated geogrid data
        updated_places=pd.concat([self.static_places, geogrid_data_gdf]).fillna(0)

        # Get the proximity scores and heatmap (before allocation of amenities)
        geo_output, pre_scores=calculate_access(
            updated_places, self.target_settings, self.reachable)

        # Create the inputs for the proximization model
        # - static norm_access
        geo_output_impact_area=geo_output.loc[geo_output['impact_area']]
        static_norm_access_cols=[col for col in geo_output_impact_area.columns if 'norm_access' in col]
        for col in static_norm_access_cols:
            updated_places.loc[updated_places['impact_area'], 'static_'+col]=list(geo_output_impact_area[col])

        for amenity in self.amenities_generative:
            updated_places['demand_for_{}'.format(amenity)]=updated_places[self.target_settings[amenity]['demand_source']].sum(axis=1)

        ind_source= (updated_places[['demand_for_{}'.format(target) for target in self.amenities_generative]].sum(axis=1)>0)
        ind_target= (updated_places['amenity_total_sqm']>0)

        source_spaces=updated_places.loc[ind_source].copy()
        target_spaces=updated_places.loc[ind_target].copy()

        # - reachable matrix, sources to targets
        reachable_s_t = self.reachable[ind_source,:][:, ind_target]
        print(reachable_s_t.shape)

        # - potential capacities of each design location for each possible target
        potential_target_capacities=np.repeat(
            [self.p_per_poi_generative[target] for target in self.amenities_generative], len(target_spaces))

        # - demand for each target at each source location
        demand_mat=updated_places.loc[
                ind_source,['demand_for_{}'.format(target) for target in self.amenities_generative]].values

        norm_access_static=updated_places.loc[
            ind_source, ['static_{}_norm_access'.format(col) for col in self.amenities_generative]
            ].fillna(0).values

        # - sqm of each design location
        sqm_target_spaces=list(updated_places.loc[ind_target,'amenity_total_sqm'])

        ind_source_in_impact_area=updated_places.loc[ind_source, 'impact_area']
        
        print('Starting proximization')
        proximizer=Proximizer(reachable_s_t, demand_mat, ind_source_in_impact_area, 
                              potential_target_capacities, self.amenities_generative, 
                              sqm_target_spaces, self.sqm_p_poi_generative, 
                              norm_access_static, integer_amenities=self.integer_amenities)
        result = proximizer.solve()
        x_final=result.x
        total_access, target_dummys, norm_capacities, norm_access=proximizer.split_design_vector(x_final)
        scores=np.array(total_access)/demand_mat.sum(axis=0)
        
        # combine pre-scores (of non-optimized targets) and and post scores (of optimized targets)
        final_scores={a: scores[i] for i, a in enumerate(self.amenities_generative)}
        for s in pre_scores:
            if s not in final_scores:
                final_scores[s]=pre_scores[s]
            
        amenity_score_dict={a: scores[i] for i, a in enumerate(self.amenities_generative)}

# TODO: the heatmap is never updated with the dynamic amenities
        non_gen_targets=[t for t in self.target_list if t not in self.amenities_generative]
        heatmap_cols=[t+'_access' for t in non_gen_targets]+['geometry']
        geo_heatmap=geo_output.loc[geo_output['impact_area'], heatmap_cols]
        geo_heatmap.columns=non_gen_targets+['geometry']
        geo_heatmap.geometry=geo_heatmap.geometry.centroid
        cs_heatmap=self.get_cs_heatmap(geo_heatmap, non_gen_targets)
        result=[{'name': '{} Prox'.format(t).title(), 
                 'value': final_scores[t],
                 'description': 'Proximity of {} from {}'.format(t, self.target_settings[t]['demand_source']),
                 'ref_value': self.baseline_scores[t]} for t in non_gen_targets]

        # lw_symm=min(final_scores['Housing'], final_scores['Jobs'])
        # avg_amenities=sum([final_scores[a] for a in self.amenities_generative])/len(self.amenities_generative)

        # result.append({'name': 'Live-Work-Symmetry',
        #     'value': lw_symm,
        #     'ref_value': self.baseline_scores['LW_Symmetry'],
        #     'viz_type': 'radar'})

        # result.append({'name': 'Amenity Access',
        #     'value': avg_amenities,
        #     'ref_value': self.baseline_scores['Amenities'],
        #     'viz_type': 'radar'})

        
        # target_dummys_final_df=pd.DataFrame(
        #     proximizer.split_by_target(self.amenities_generative, target_dummys, proximizer.n_design))
        # print(target_dummys_final_df.apply(np.floor))
        
        # new_amenities=target_dummys_final_df.apply(np.floor).sum(axis=0).to_dict()

        print('Time taken: {}'.format(datetime.datetime.now()-start_time))  
        return {'heatmap':cs_heatmap,'numeric':result}

