# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Validation script lib_grid.py
# =====================================================================================================================

import numpy as np
from lib_grid import grid_properties, polar_grid_properties


if __name__ == '__main__':

    print('Validating grid.py !!!!!! ........')

    GridPolar = polar_grid_properties(
        min_range = 0.0, 
        max_range = 10.0, 
        min_azimuth = -180.0, 
        max_azimuth = 180.0, 
        range_res = 0.5, 
        azimuth_res = 0.5,
        theta_res_min = 0.2,
        theta_res_max = 5.0
    )

    GridCartesian = grid_properties(
        min_x = -10.0, 
        max_x = 10.0, 
        min_y = -5.0, 
        max_y = 5.0, 
        dx = 1, 
        dy = 1
    )

    r_coord, a_coord = GridPolar.compute_ra_coordinates_from_ra_ids(GridPolar.range_ids, GridPolar.azimuth_ids)
    print('y coord: ', a_coord); print('-' * 100)


    print('max range              : ', GridPolar.min_range)
    print('min range              : ', GridPolar.max_range)
    print('max azimuth            : ', GridPolar.min_azimuth)
    print('min azimuth            : ', GridPolar.max_azimuth)
    print('num range coordinates  : ', GridPolar.num_cells_range)
    print('num azimuth coordinates: ', GridPolar.num_cells_azimuth)
    print('range ids: ', GridPolar.range_ids)
    print('-' * 100)
    print('azimuth ids: ', GridPolar.azimuth_ids)
    print('-' * 100)
    print('range coord: ', GridPolar.range_coord)
    print('-' * 100)
    print('azimuth coord: ', GridPolar.azimuth_coord)
    print('-' * 100)
    print('azimuth cell resolutions: ', GridPolar.theta_res)
    print('azimuth cell boundary upper: ', GridPolar.theta_upper)
    print('azimuth cell boundary lower: ', GridPolar.theta_lower)
    



    print('=' * 100)
    print('max x            : ', GridCartesian.max_x)
    print('min x            : ', GridCartesian.min_x)
    print('max y            : ', GridCartesian.max_y)
    print('min y            : ', GridCartesian.min_y)
    print('num x coordinates: ', GridCartesian.num_cells_x)
    print('num y coordinates: ', GridCartesian.num_cells_y)
    print('total num cells  : ', GridCartesian.num_cells)
    print('-' * 100)
    print('x ids: ', GridCartesian.x_ids)
    print('-' * 100)
    print('y ids: ', GridCartesian.y_ids)
    print('-' * 100)
    print('x coord: ', GridCartesian.x_coord)
    print('-' * 100)
    print('y coord: ', GridCartesian.y_coord)
    print('scalar ids: ', GridCartesian.cell_ids)



    col_idx, row_idx = np.meshgrid(np.arange(GridCartesian.num_cells_y), np.arange(GridCartesian.num_cells_x))
    xy_ids = np.reshape(np.stack([row_idx, col_idx], axis=-1), (-1, 2))

    print('=' * 100)
    x_coord, y_coord = GridCartesian.compute_xy_coordinates_from_xy_ids(GridCartesian.x_ids, GridCartesian.y_ids)
    x_ids, y_ids = GridCartesian.compute_xy_ids_from_xy_coordinates(x_coord, y_coord)
    scalar_ids = GridCartesian.compute_scalar_ids_from_xy_ids(xy_ids[:,0], xy_ids[:,1])
    x1_ids, y1_ids = GridCartesian.compute_xy_ids_from_scalar_ids(scalar_ids)
    print('x coord: ', x_coord)
    print('y coord: ', y_coord); print('-' * 100)
    print('x ids  : ', x_ids)
    print('y ids  : ', y_ids); print('-' * 100)
    print('scalar_ids  : ', scalar_ids); print('-' * 100)
    print('x ids  : ', x1_ids)
    print('y ids  : ', y1_ids); print('-' * 100)



    
    y_coord, x_coord = np.meshgrid(GridCartesian.y_coord, GridCartesian.x_coord)
    xy_coord = np.reshape(np.stack([x_coord, y_coord], axis=-1), (-1, 2))

    print('=' * 100)
    scalar_ids = GridCartesian.compute_scalar_ids_from_xy_coordinates(xy_coord[:,0], xy_coord[:,1])
    x_coord, y_coord = GridCartesian.compute_xy_coordinates_from_scalar_ids(scalar_ids)
    print('scalar_ids  : ', scalar_ids); print('-' * 100)
    print('x_coord  : ', x_coord)
    print('y_coord  : ', y_coord); print('-' * 100)


    