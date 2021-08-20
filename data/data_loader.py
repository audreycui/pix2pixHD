
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader, StyleGANDatasetDataLoader
    if opt.generated: 
        data_loader = StyleGANDatasetDataLoader()
    else: 
        data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

