from src.graph.positions import coordinate_to_position as pos


def test_high_line_centre_backs_stay_centre_backs():
    assert pos(39.2, 33.0, "D") == "RCB"   # Saliba-like
    assert pos(38.4, 65.1, "D") == "LCB"


def test_full_backs():
    assert pos(52, 21, "D") == "RB"
    assert pos(44, 76, "D") == "LB"
    assert pos(41, 20, "D") == "RB"       # deep but wide → still a full-back


def test_base_code_decides_class():
    assert pos(45, 50, "M") == "CDM"      # deep midfielder never becomes a CB
    assert pos(66, 20, "F") == "RW"
    assert pos(None, None, "G") == "GK"
