"""
Explanation Generator
Never returns naked labels - always provides reasoning
Generates human-readable forensic reports
"""

def score_to_level(score):
    """Convert score to severity level"""
    if score < 0.3:
        return "Low", "low"
    elif score < 0.6:
        return "Moderate", "medium"
    else:
        return "High", "high"

def generate_stylometry_explanation(score):
    """Explain stylometry findings"""
    level, severity = score_to_level(score)
    
    explanations = {
        "low": {
            "text": "Stylometric patterns appear natural",
            "detail": "Writing behavior shows expected human variation in sentence structure, vocabulary, and linguistic quirks."
        },
        "medium": {
            "text": "Moderate stylometric deviation detected",
            "detail": "Some linguistic patterns differ from typical human variation. Sentence length variance or vocabulary usage shows normalization."
        },
        "high": {
            "text": "High stylometric deviation detected",
            "detail": "Writing behavior shows unusual consistency or normalization patterns characteristic of AI editing. Reduced variance in sentence structure and unnatural vocabulary distribution detected."
        }
    }
    
    return explanations[severity]

def generate_ai_text_explanation(score):
    """Explain AI text detection findings"""
    level, severity = score_to_level(score)
    
    explanations = {
        "low": {
            "text": "Low AI language probability",
            "detail": "Text shows natural linguistic patterns with expected human inconsistencies."
        },
        "medium": {
            "text": "Moderate AI language probability",
            "detail": "Some linguistic markers suggest possible AI editing. Text shows increased fluency or predictability."
        },
        "high": {
            "text": "High AI language probability",
            "detail": "Text shows increased predictability and fluency normalization characteristic of AI-generated content. Formal connectors and structured phrasing detected."
        }
    }
    
    return explanations[severity]

def generate_image_explanation(scores):
    """Explain image forensics findings"""
    img_score = scores['image_forensics']
    level, severity = score_to_level(img_score)
    
    base_explanations = {
        "low": {
            "text": "Image properties appear authentic",
            "detail": "Noise patterns, stroke textures, and paper characteristics consistent with real camera capture."
        },
        "medium": {
            "text": "Image shows some digital artifacts",
            "detail": "Minor inconsistencies detected in noise patterns or stroke behavior."
        },
        "high": {
            "text": "Image forensics show synthetic indicators",
            "detail": "Noise patterns and stroke textures inconsistent with camera capture."
        }
    }
    
    explanation = base_explanations[severity]
    
    # Add specific module details
    details = []
    
    if scores['noise'] > 0.5:
        details.append("Noise pattern analysis: Spatial variance and frequency spectrum inconsistent with camera sensor")
    
    if scores['stroke'] > 0.5:
        details.append("Stroke texture analysis: Width variance and entropy lower than expected for natural ink")
    
    if scores['paper'] > 0.5:
        details.append("Paper texture analysis: Background uniformity suggests digital generation")
    
    if details:
        explanation['detail'] += " " + " | ".join(details)
    
    return explanation

def generate_recommendation(verdict, confidence, text_length):
    """Generate actionable recommendations"""
    recommendations = []
    
    if verdict == "AI-Assisted Likely":
        recommendations.append("Consider requesting original handwritten document for verification")
        recommendations.append("Review document context and submission circumstances")
        recommendations.append("Cross-reference with known authentic samples if available")
    
    elif verdict == "Suspicious":
        recommendations.append("Document requires additional verification")
        recommendations.append("Consider combining with other authentication methods")
        recommendations.append("Review for consistency with other submitted materials")
    
    else:  # Likely Human
        recommendations.append("Document shows characteristics of authentic handwriting")
        recommendations.append("No strong synthetic indicators detected")
    
    # Add warnings about limitations
    if text_length < 50:
        recommendations.append("⚠️ Short text limits stylometry accuracy - interpret with caution")
    
    if confidence < 60:
        recommendations.append("⚠️ Moderate confidence - consider additional evidence")
    
    return recommendations

def generate_technical_summary(scores):
    """Generate technical details for experts"""
    return {
        "fusion_formula": "0.35×Stylometry + 0.30×AI_Text + 0.25×Image + 0.10×Adjustments",
        "final_score": f"{scores['final']:.3f}",
        "decision_thresholds": {
            "human": "< 0.40",
            "suspicious": "0.40 - 0.65",
            "ai_assisted": "> 0.65"
        },
        "module_scores": {
            "stylometry": f"{scores['stylometry']:.3f}",
            "ai_text": f"{scores['ai_text']:.3f}",
            "image_forensics": f"{scores['image_forensics']:.3f}",
            "noise": f"{scores['noise']:.3f}",
            "stroke": f"{scores['stroke']:.3f}",
            "paper": f"{scores['paper']:.3f}"
        }
    }

def generate_explanation(verdict, confidence, scores, raw_text):
    """
    Main explanation generator
    Never returns naked verdict - always includes reasoning
    
    Returns complete forensic report
    """
    text_length = len(raw_text) if raw_text else 0
    
    # Generate component explanations
    stylometry_exp = generate_stylometry_explanation(scores['stylometry'])
    ai_text_exp = generate_ai_text_explanation(scores['ai_text'])
    image_exp = generate_image_explanation(scores)
    
    # Compile findings
    findings = [
        {
            "module": "Stylometry Analysis",
            "severity": score_to_level(scores['stylometry'])[1],
            "finding": stylometry_exp['text'],
            "detail": stylometry_exp['detail']
        },
        {
            "module": "AI Text Detection",
            "severity": score_to_level(scores['ai_text'])[1],
            "finding": ai_text_exp['text'],
            "detail": ai_text_exp['detail']
        },
        {
            "module": "Image Forensics",
            "severity": score_to_level(scores['image_forensics'])[1],
            "finding": image_exp['text'],
            "detail": image_exp['detail']
        }
    ]
    
    # Generate recommendations
    recommendations = generate_recommendation(verdict, confidence, text_length)
    
    # Technical summary
    technical = generate_technical_summary(scores)
    
    # Compile full report
    report = {
        "verdict": verdict,
        "confidence": confidence,
        "findings": findings,
        "recommendations": recommendations,
        "technical_summary": technical,
        "warnings": []
    }
    
    # Add warnings
    if text_length < 50:
        report['warnings'].append("Very short text - stylometry accuracy reduced")
    
    if text_length < 20:
        report['warnings'].append("Extremely short text - results highly uncertain")
    
    if scores.get('adjustments', 0) > 0.15:
        report['warnings'].append("Quality penalties applied - input may be suboptimal")
    
    return report

def format_report_text(report):
    """
    Format report as readable text (for console/logs)
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AI HANDWRITING FORENSICS REPORT")
    lines.append("=" * 60)
    lines.append(f"\nVERDICT: {report['verdict']}")
    lines.append(f"CONFIDENCE: {report['confidence']}%")
    lines.append("\nFINDINGS:")
    lines.append("-" * 60)
    
    for finding in report['findings']:
        lines.append(f"\n{finding['module']} [{finding['severity'].upper()}]")
        lines.append(f"  • {finding['finding']}")
        lines.append(f"  └ {finding['detail']}")
    
    lines.append("\n" + "-" * 60)
    lines.append("RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        lines.append(f"  {i}. {rec}")
    
    if report['warnings']:
        lines.append("\n" + "-" * 60)
        lines.append("WARNINGS:")
        for warning in report['warnings']:
            lines.append(f"  ⚠️  {warning}")
    
    lines.append("\n" + "=" * 60)
    lines.append("Technical Summary:")
    lines.append(f"  Formula: {report['technical_summary']['fusion_formula']}")
    lines.append(f"  Final Score: {report['technical_summary']['final_score']}")
    lines.append("=" * 60)
    
    return "\n".join(lines)