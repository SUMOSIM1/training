import training.simrunner as sr
import training.simrunner_core as src


def test_format_command_start():
    data = sr._format_command(sr.StartCommand())
    assert data == "A|"


def test_format_diff_drive_1():
    r1 = sr.DiffDriveValues(1.5, 4.3)
    r2 = sr.DiffDriveValues(-1.2, 44.0)
    data = sr._format_command(sr.DiffDriveCommand(r1, r2, 123, True))
    assert data == "C|4.3000;1.5000#44.0000;-1.2000#123#1"


def test_format_diff_drive_2():
    r1 = sr.DiffDriveValues(1.5, 4.3)
    r2 = sr.DiffDriveValues(-1.2, 44.0)
    data = sr._format_command(sr.DiffDriveCommand(r1, r2, 111, False))
    assert data == "C|4.3000;1.5000#44.0000;-1.2000#111#0"


def test_parse_sensor():
    expected = sr.CombiSensorCommand(
        robot1_sensor=sr.CombiSensorDto(
            pos_dir=src.PosDir(-80.0, 0.0, 0.1396),
            combi_sensor=sr.CombiSensor(
                left_distance=357.6570,
                front_distance=506.4382,
                right_distance=398.3340,
                opponent_in_sector=sr.SectorName.RIGHT,
            ),
        ),
        robot2_sensor=sr.CombiSensorDto(
            pos_dir=src.PosDir(-80.0, 0.0, 0.0),
            combi_sensor=sr.CombiSensor(
                left_distance=385.7849,
                front_distance=462.9790,
                right_distance=401.1228,
                opponent_in_sector=sr.SectorName.CENTER,
            ),
        ),
    )
    data = (
        "B|-80.0000;0.0000;0.1396;357.6570;506.4382;398.3340;RIGHT#"
        "-80.0000;0.0000;0.0000;385.7849;462.9790;401.1228;CENTER"
    )
    cmd = sr._parse_command(data)
    assert cmd == expected


def test_parse_finished_ok():
    data = "D|#winner!true"
    cmd = sr._parse_command(data)
    assert cmd == sr.FinishedOkCommand([], [("winner", "true")])
