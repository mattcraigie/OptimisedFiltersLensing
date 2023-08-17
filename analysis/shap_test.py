import shap
import torch
import matplotlib.pyplot as plt

# load the deep regression model


def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


def load_data(data_path):
    data = torch.load(data_path)
    return data


def get_shap_values(model, data, subset):
    if subset is not None:
        data = data[:subset]
    e = shap.DeepExplainer(model, data)
    shap_values = e.shap_values(data)
    return shap_values


def get_summary_plot(shap_values, data, save_path=None):
    shap.summary_plot(shap_values, data, show=False)
    if save_path is not None:
        plt.savefig(save_path)


def produce_summary_plot(model_path, data_path, save_path=None, subset=None):
    model = load_model(model_path)
    data = load_data(data_path)
    shap_values = get_shap_values(model, data, subset)
    get_summary_plot(shap_values, data, save_path)



