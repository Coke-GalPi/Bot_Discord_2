import os
import clip
import torch
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
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

# Cargar los datasets CIFAR10 y CIFAR100
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True)
text_inputs10 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)

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

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)

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

@bot.command(name='Image')
async def image(ctx):
    print('Image!')
    if len(ctx.message.attachments) == 0:
        await ctx.send('No se ha encontrado ninguna imagen adjunta.')
        return

    image_url = ctx.message.attachments[0].url

    # Descargar y procesar la imagen en paralelo
    image, image_input = await asyncio.to_thread(download_and_preprocess_image, image_url)

    # Calcular características para CIFAR10 y CIFAR100 en paralelo
    results = await asyncio.gather(
        asyncio.to_thread(encode_image_text, image_input, text_inputs10),
        asyncio.to_thread(encode_image_text, image_input, text_inputs100)
    )
    image_features10, text_features10 = results[0]
    image_features100, text_features100 = results[1]

    # Calcular similitud en paralelo
    similarities = await asyncio.gather(
        asyncio.to_thread(calculate_similarity, image_features10, text_features10),
        asyncio.to_thread(calculate_similarity, image_features100, text_features100)
    )
    values10, indices10 = similarities[0]
    values100, indices100 = similarities[1]

    # Seleccionar el dataset con la mayor similitud
    if values10.max() >= values100.max():
        values = values10
        indices = indices10
        classes = cifar10.classes
    else:
        values = values100
        indices = indices100
        classes = cifar100.classes

    # Enviar los resultados al canal de Discord
    print('Enviar los resultados al canal de Discord')
    if values[0].item() >= 0.9:
        await ctx.send(f"Prediction: {classes[indices[0]]}")
    elif 0.75 <= values[0].item() < 0.9:
        results = "\n".join([f"{classes[index]}: {100 * value.item():.2f}%" for value, index in zip(values, indices)])
        await ctx.send(f"Top predictions:\n{results}")
    else:
        await ctx.send("Cannot be predicted.")

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