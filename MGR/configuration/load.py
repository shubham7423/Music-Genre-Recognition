import yaml


def load_configurations():
    """Function to load configurations from yaml file

    Returns:
    ________
    config: dict
        Dictionary containing configurations
    """
    with open('mgr/configuration/configuration.yaml', 'r') as file:
        return yaml.load(file)
