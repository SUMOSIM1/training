import training.value_mapping as vm


def main():
    max_speed = 7.0
    max_speed_index = 5
    t1 = vm._diff_drive_tuple(max_speed, max_speed_index)

    for j, (a, b) in enumerate(t1):
        print(
            f"           ({max_speed:.2f}, {max_speed_index}, {j}, {a:.2f}, {b:.2f}),"
        )
