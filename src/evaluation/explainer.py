import shap
import matplotlib.pyplot as plt

def generate_shap_explanation(model, X_instance):
    """
    Generates a SHAP waterfall plot for a single loan application.
    """
    explainer = shap.TreeExplainer(model)
    # Generate SHAP values for the instance
    shap_values = explainer(X_instance, check_additivity=False)
    
    # Create a matplotlib figure
    fig = plt.figure(figsize=(10, 6))
    
    # If the model is a classifier, shap_values might have 2 dimensions (0=Paid, 1=Default)
    # We plot the impact on 'Default' (index 1)
    if len(shap_values.shape) > 2:
        shap.plots.waterfall(shap_values[0][:, 1], show=False)
    else:
        shap.plots.waterfall(shap_values[0], show=False)
    
    plt.tight_layout()
    return fig # Crucial: Return the figure object