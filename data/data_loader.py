
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader, StyleGANDatasetDataLoader
    data_loader = StyleGANDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

