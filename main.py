import os, torch, torchvision
from src.data import blend
from PIL import Image
from src.models import PretrainedUNet

def main():
    origin_filename = '/gris/gris-f/homestud/aranem/lung-segmentation/1.3.51.0.7.597266080.44770.52800.43566.28465.1231.17369.jpg'
    result = '/gris/gris-f/homestud/aranem/lung-segmentation/seg.jpg'
    result2 = '/gris/gris-f/homestud/aranem/lung-segmentation/joined.jpg'
    device = 'cuda:0'
    unet = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
    )

    model_name = "unet-6v.pt"
    unet.load_state_dict(torch.load(os.path.join('models', model_name), map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval()

    print("Load image..")
    origin = Image.open(origin_filename).convert("P")
    origin = torchvision.transforms.functional.resize(origin, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

    print("Predict..")
    with torch.no_grad():
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = unet(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
        
        origin = origin[0].detach().cpu()
        out = out[0].detach().cpu()
        
    img =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([out.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))

    img.save(result)

    img = blend(origin, out)
    img.save(result2)
    
if __name__ == "__main__":
    main()