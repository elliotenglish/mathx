def ray_intersect_aabb(ray_origin, ray_direction, aabb_min, aabb_max):
    """
    Checks if a ray intersects an axis-aligned bounding box (AABB).

    Args:
        ray_origin: numpy array of shape (3,) representing the ray's origin.
        ray_direction: numpy array of shape (3,) representing the ray's direction (normalized).
        aabb_min: numpy array of shape (3,) representing the AABB's minimum corner.
        aabb_max: numpy array of shape (3,) representing the AABB's maximum corner.

    Returns:
        A tuple containing:
        - True if the ray intersects the AABB, False otherwise.
        - If there is an intersection, the parameter t at which the ray enters the AABB.
          If there is no intersection, returns infinity.
    """
    t_min = (aabb_min - ray_origin) / ray_direction
    t_max = (aabb_max - ray_origin) / ray_direction

    t_enter = max(np.minimum(t_min, t_max))
    t_exit = min(np.maximum(t_min, t_max))

    return t_enter<=t_exit, t_enter, t_exit

    # if t_enter > 0 and t_enter <= t_exit:
    #     return True, t_enter
    # else:
    #     return False, float('inf')
