import logging
import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import ParseMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN")
huggingface_api_token = os.getenv("huggingface_api_token")
# Telegram Bot Token
token = 'TOKEN'

# Hugging Face Model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # You can change to other models like "gpt2-medium", "gpt2-large", etc.

# Initialize the Hugging Face model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=huggingface_api_token)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Function to generate response using Hugging Face LLM
async def generate_response(input_text: str) -> str:
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Command handler for /start command
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer("Hi! I am your Telegram Bot. Send me a message, and I will generate a response for you!")

# Message handler
@dp.message_handler(content_types=['text'])
async def handle_message(message: types.Message):
    user_input = message.text
    response = await generate_response(user_input)
    await message.reply(response, parse_mode=ParseMode.HTML)

# Main function to run the bot
async def main():
    # Start the bot
    await dp.start_polling()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
