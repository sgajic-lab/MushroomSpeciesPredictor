import torch
from PIL import Image
from model import ConvNet # Imports the class from your other file!
from torchvision.transforms import v2


# all classes being tested for
classes = ['Agaricus augustus', 'Agaricus xanthodermus', 'Amanita amerirubescens', 'Amanita augusta', 'Amanita brunnescens', 'Amanita calyptroderma', 'Amanita citrina', 'Amanita flavoconia', 'Amanita muscaria', 'Amanita pantherina', 'Amanita persicina', 'Amanita phalloides', 'Amanita rubescens', 'Amanita velosa', 'Apioperdon pyriforme', 'Armillaria borealis', 'Armillaria mellea', 'Armillaria tabescens', 'Artomyces pyxidatus', 'Bjerkandera adusta', 'Bolbitius titubans', 'Boletus edulis', 'Boletus pallidus', 'Boletus reticulatus', 'Boletus rex-veris', 'Calocera viscosa', 'Calycina citrina', 'Cantharellus californicus', 'Cantharellus cibarius', 'Cantharellus cinnabarinus', 'Cerioporus squamosus', 'Cetraria islandica', 'Chlorociboria aeruginascens', 'Chlorophyllum brunneum', 'Chlorophyllum molybdites', 'Chondrostereum purpureum', 'Cladonia fimbriata', 'Cladonia rangiferina', 'Cladonia stellaris', 'Clitocybe nebularis', 'Clitocybe nuda', 'Coltricia perennis', 'Coprinellus disseminatus', 'Coprinellus micaceus', 'Coprinopsis atramentaria', 'Coprinopsis lagopus', 'Coprinus comatus', 'Crucibulum laeve', 'Cryptoporus volvatus', 'Daedaleopsis confragosa', 'Daedaleopsis tricolor', 'Entoloma abortivum', 'Evernia mesomorpha', 'Evernia prunastri', 'Flammulina velutipes', 'Fomes fomentarius', 'Fomitopsis betulina', 'Fomitopsis mounceae', 'Fomitopsis pinicola', 'Galerina marginata', 'Ganoderma applanatum', 'Ganoderma curtisii', 'Ganoderma oregonense', 'Ganoderma tsugae', 'Gliophorus psittacinus', 'Gloeophyllum sepiarium', 'Graphis scripta', 'Grifola frondosa', 'Gymnopilus luteofolius', 'Gyromitra esculenta', 'Gyromitra gigas', 'Gyromitra infula', 'Hericium coralloides', 'Hericium erinaceus', 'Hygrophoropsis aurantiaca', 'Hypholoma fasciculare']



def predict_mushroom(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_demo = ConvNet(num_classes=76).to(device)
    model_demo.load_state_dict(torch.load("mushroom_model.pt", map_location=device, weights_only=True))
    model_demo.eval()

    # same transforms as val
    demo_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # process and format input image --> model is expecting tensors
    # making sure there are three color channels that the model expects
    image = Image.open(image_path).convert("RGB")
    input_tensor = demo_transforms(image)
    # creates a batch of one
    input_batch = input_tensor.unsqueeze(0).to(device)

    # make the prediction
    with torch.no_grad():
        outputs = model_demo(input_batch)

        # .max returns the maximum value and the index of the maximum value
        # _, discards the first number and focuses only on the index so we can look at the index to be able to identify the class
        _, predicted_idx = outputs.max(1)
        final_index = predicted_idx.item()
    print(f"The model predicts: {classes[final_index]}")

if __name__ == "__main__":
    predict_mushroom("Amanita Amerirubescens.jpg")