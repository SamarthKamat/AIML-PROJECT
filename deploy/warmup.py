"""Pre-compute predictions and graph metrics so the first page load is fast."""
from backend.app import create_app, services

app = create_app()
with app.app_context():
    services.ensure_predictions()
    services.graph_metrics_df()
print("warm-up complete")
