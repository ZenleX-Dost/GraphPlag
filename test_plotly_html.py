#!/usr/bin/env python
"""Quick test to see if Plotly HTML embedding works"""
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=65,
    number={'suffix': "%", 'font': {'size': 48}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#28a745"},
        'steps': [
            {'range': [0, 50], 'color': '#d4edda'},
            {'range': [50, 70], 'color': '#fff3cd'},
            {'range': [70, 100], 'color': '#ffb8c8'}
        ]
    }
))

fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

html_output = fig.to_html(include_plotlyjs='cdn', div_id="test-gauge")

print("HTML length:", len(html_output))
print("First 500 chars:")
print(html_output[:500])
print("\nHTML contains <script>:", '<script' in html_output)
print("HTML contains <div>:", '<div' in html_output)

# Test with full page
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Test</title>
</head>
<body>
    <h1>Test Plot</h1>
    {html_output}
</body>
</html>
"""

with open('/tmp/test_plot.html', 'w') as f:
    f.write(full_html)

print("\nTest HTML saved to /tmp/test_plot.html")
