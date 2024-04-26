def calculate_eer(far, frr):
    min_diff = float('inf')
    eer = None
    for i in range(len(far)):
        diff = abs(far[i] - frr[i])
        if diff < min_diff:
            min_diff = diff
            eer = (far[i] + frr[i]) / 2  # EER is the average of FAR and FRR at the closest point
    return eer

def main():
    orbfrr = [0.305, 0.28, 0.32, 0.335, 0.31]
    orbfar = [0.053, 0.0544, 0.052, 0.0532, 0.0515]
    print(f"ORB: {calculate_eer(orbfar, orbfrr):.4f}")

    pilfrr = [0.0275, 0.035, 0.035, 0.0375, 0.03]
    pilfar = [0.1263, 0.1253, 0.1214, 0.1274, 0.1179]
    print(f"PIL: {calculate_eer(pilfar, pilfrr):.4f}")

    rmsfrr = [0.0925, 0.085, 0.0825, 0.0675, 0.08]
    rmsfar = [0.1075, 0.1175, 0.1161, 0.1154, 0.1103]
    print(f"RMS: {calculate_eer(rmsfar, rmsfrr):.4f}")

    hybridfrr = [0.07, 0.065, 0.075, 0.045, 0.0625]
    hybridfar = [0.1172, 0.1168, 0.1114, 0.1136, 0.1224]
    print(f"HYBRID: {calculate_eer(hybridfar, hybridfrr):.4f}")

main()