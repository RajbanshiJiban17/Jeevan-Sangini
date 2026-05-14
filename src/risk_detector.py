def detect_risk(report_text):

    report = report_text.lower()

    risks = []

    if "hb" in report and "9.5" in report:
        risks.append("🚨 Anemia Risk")

    if "blood pressure" in report:
        risks.append("⚠️ Blood Pressure Monitoring Needed")

    if "fever" in report:
        risks.append("⚠️ Infection Risk")

    return risks