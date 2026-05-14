def detect_risk(report_text):

    report = report_text.lower()

    risks = []

    # =====================================
    # ANEMIA DETECTION
    # =====================================
    if "hb" in report or "hemoglobin" in report:

        if "9" in report or "9.5" in report or "low hb" in report:
            risks.append({
                "type": "🚨 Anemia Risk",
                "level": "HIGH",
                "action": "Iron supplement + doctor consultation"
            })

        elif "11" in report:
            risks.append({
                "type": "ℹ️ Borderline Hemoglobin",
                "level": "MEDIUM",
                "action": "Diet improvement recommended"
            })

    # =====================================
    # BLOOD PRESSURE
    # =====================================
    if "blood pressure" in report or "bp" in report:

        if "140" in report or "high" in report:
            risks.append({
                "type": "⚠️ High Blood Pressure",
                "level": "HIGH",
                "action": "Immediate medical monitoring needed"
            })

        else:
            risks.append({
                "type": "📊 Blood Pressure Monitoring",
                "level": "LOW",
                "action": "Regular check-up advised"
            })

    # =====================================
    # FEVER / INFECTION
    # =====================================
    if "fever" in report or "temperature" in report:

        risks.append({
            "type": "⚠️ Possible Infection",
            "level": "MEDIUM",
            "action": "Monitor symptoms, consult doctor if persists"
        })

    # =====================================
    # PREGNANCY WARNING SIGNALS
    # =====================================
    if "pregnancy" in report:

        if "pain" in report or "bleeding" in report:
            risks.append({
                "type": "🚨 Pregnancy Complication Risk",
                "level": "HIGH",
                "action": "Immediate hospital visit required"
            })

    # =====================================
    # IF NOTHING FOUND
    # =====================================
    if len(risks) == 0:
        risks.append({
            "type": "✅ No Major Risk Detected",
            "level": "NORMAL",
            "action": "Routine check-up only"
        })

    return risks