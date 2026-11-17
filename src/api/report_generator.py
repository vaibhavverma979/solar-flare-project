import tempfile
import os
import re
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

async def generate_advisory_pdf(report_text: str, current_telemetry: dict, forecast_metrics: dict) -> str:
    """Generates an ISRO Space Weather Advisory PDF and returns the file path."""
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="isro_advisory_")
    os.close(fd)
    
    doc = SimpleDocTemplate(temp_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=14,
        textColor=colors.darkblue
    )
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    elements = []
    
    # Header
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elements.append(Paragraph("ISRO ADITYA-L1 SPACE WEATHER ADVISORY REPORT", title_style))
    elements.append(Paragraph(f"Generated at: {timestamp}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    # In case the report_text has newlines, split them into multiple paragraphs
    for para in report_text.split('\n'):
        if para.strip():
            cleaned = para.strip()
            
            # Headings
            if cleaned.startswith('### '):
                cleaned = f'<b><font size="14">{cleaned[4:]}</font></b>'
            elif cleaned.startswith('#### '):
                cleaned = f'<b><font size="12">{cleaned[5:]}</font></b>'
            elif cleaned.startswith('## '):
                cleaned = f'<b><font size="16">{cleaned[3:]}</font></b>'
            elif cleaned.startswith('# '):
                cleaned = f'<b><font size="18">{cleaned[2:]}</font></b>'
            
            # Bold
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cleaned)
            
            # Bullets
            if cleaned.startswith('- '):
                cleaned = f'&bull; {cleaned[2:]}'
            elif cleaned.startswith('* '):
                cleaned = f'&bull; {cleaned[2:]}'
                
            elements.append(Paragraph(cleaned, normal_style))
    elements.append(Spacer(1, 20))
    
    # Current Telemetry
    elements.append(Paragraph("Current Telemetry", heading_style))
    
    telemetry_data = [
        ["Metric", "Value"],
        ["SoLEXS Flux (max)", str(current_telemetry.get("solexs_flux_max", "N/A"))],
        ["HEL1OS Flux (max)", str(current_telemetry.get("hel1os_flux_max", "N/A"))],
        ["MAG Bz (avg)", str(current_telemetry.get("mag_b_total_avg", "N/A"))],
        ["ASPEX Proton Density (avg)", str(current_telemetry.get("aspex_proton_density_avg", "N/A"))],
        ["ASPEX Solar Wind Velocity (avg)", str(current_telemetry.get("aspex_solar_wind_velocity_avg", "N/A"))]
    ]
    
    telemetry_table = Table(telemetry_data, colWidths=[200, 200])
    telemetry_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(telemetry_table)
    elements.append(Spacer(1, 20))
    
    # Forecast Metrics
    elements.append(Paragraph("Forecast", heading_style))
    
    forecast_data = [
        ["Metric", "Value"],
        ["30-min Flare Probability", str(forecast_metrics.get("flare_probability", "N/A"))],
        ["Predicted Dst", str(forecast_metrics.get("predicted_dst", "N/A"))],
        ["Predicted Kp Index", str(forecast_metrics.get("predicted_kp", "N/A"))]
    ]
    
    forecast_table = Table(forecast_data, colWidths=[200, 200])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(forecast_table)
    
    # Build PDF
    doc.build(elements)
    
    return temp_path
