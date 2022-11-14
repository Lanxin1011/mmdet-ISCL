from shapely.validation import explain_validity


def single_valid_polygon(polygon):
    """check the validation of polygon

    Args:
        polygon (Polygon): input polygon

    Returns:
        bool: the flag of validation
    """
    if not polygon.is_valid:
        return False
    elif 'Self-intersection' in explain_validity(polygon):
        print("This polygon is self intersection: ", polygon)
        return False
    elif polygon.geom_type not in ['Polygon', 'MultiPolygon']:
        print("This polygon is a error type: ", type(polygon))
        return False
    else:
        return True