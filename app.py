# Librerias Modelo Clip
import os
import clip
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import CIFAR100
import requests
from io import BytesIO

# Librerias Discord
import discord
from discord.ext import commands
import asyncio

# Librerias para cargar variables de entorno
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
token_key = os.getenv('DISCORD_TOKEN')
channel_ids = os.getenv('DISCORD_CHANNEL_IDS').split(',')

# Convertir cada ID de canal a un número entero
channel_ids = [int(channel_id.strip()) for channel_id in channel_ids]

# Configurar los intents necesarios
intents = discord.Intents.default()
intents.message_content = True

# Crear una instancia del bot con un prefijo de comando
bot = commands.Bot(command_prefix='!', intents=intents)

# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)

# Cargar los datasets CIFAR100
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True)
text_inputs100 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')
    for channel_id in channel_ids:
        channel = bot.get_channel(channel_id)
        if channel:
            await channel.send(
                'Hello, I am Bot_Clip\n'
                'I am ready to help you with the CLIP model!\n'
                'Type !Help to see the available commands.\n'
            )
        else:
            print(f"Channel with ID {channel_id} not found")

@bot.command(name='Hello')
async def hello(ctx):
    print('Hello, world!')
    await ctx.send('Hello, world!')

@bot.command(name='Thanks')
async def thanks(ctx):
    print('Thanks!')
    await ctx.send('You are welcome!')

@bot.command(name='Help')
async def help(ctx):
    print("Help")
    await ctx.send(
        'Welcome to ClipBot!\n'
        'This bot allows you to perform the following actions:\n'
        '1. !Hello: Greet the bot\n'
        '2. !Image: Recognize the content of an image\n'
            '\tStep 1: Type the command "!image"\n'
            '\tStep 2: Add the image you want to analyze\n'
            '\tStep 3: Press enter to send the image\n'
            '\tStep 4: Wait for the bot\'s response\n'
        '3. !Thanks: Thank the bot\n'
    )

@bot.command(name='Image', aliases=['image'])
async def image(ctx):
    print('Image!')
    if len(ctx.message.attachments) == 0:
        await ctx.send('No images were found attached.')
        return

    image_url = ctx.message.attachments[0].url
    
    # Descargar el archivo y comprobar si es una imagen
    is_image = await asyncio.to_thread(check_image_type, image_url)
    if not is_image:
        await ctx.send('The attached file is not a valid image.')
        return
    
    # Descargar y procesar la imagen
    image, image_input = await asyncio.to_thread(download_and_preprocess_image, image_url)
    
    # Calcular características para CIFRAR100
    results = await asyncio.to_thread(encode_image_text, image_input, text_inputs100)
    image_features100, text_features100 = results
    
    # Calcular similitud
    similarities = await asyncio.to_thread(calculate_similarity, image_features100, text_features100)
    values100, indices100 = similarities
    
    # Obtener las clases de CIFAR100
    classes = cifar100.classes
    
    # Enviar la respuesta
    print('Send the results to the Discord channel')
    if values100[0] >= 0.9:
        await ctx.send(f"Image: {image_url} \nPrediction: {classes[indices100[0]]}")
    elif 0.9 > values100[0] >= 0.5:
        results = "\n".join([f"{classes[index]}: {100 * value.item():.2f}%" for value, index in zip(values100, indices100)])
        await ctx.send(f"Image: {image_url} \nTop predictions:\n{results}")
    else:
        await ctx.send(f"Image: {image_url} \nCannot be predicted.")

def check_image_type(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image.verify()  # Verifica que es una imagen
        return True
    except (UnidentifiedImageError, requests.exceptions.RequestException):
        return False

def download_and_preprocess_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    # Preprocesar la imagen
    image_input = preprocess(image).unsqueeze(0).to(device)
    return image, image_input

def encode_image_text(image_input, text_inputs):
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    return image_features, text_features

def calculate_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)
    return values, indices

# Ejecutar el bot con el token
bot.run(token_key)