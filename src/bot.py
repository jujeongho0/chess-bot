import io
import os
from collections import defaultdict

import chess
import discord
import torch
from discord import app_commands
from discord.ui import Button, View
from PIL import Image
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from generator import Generator


torch.manual_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)
configuration = GPT2Config.from_pretrained("gpt2.pth", output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2.pth", config=configuration)
model.resize_token_embeddings(len(tokenizer))


class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync(guild=discord.Object(id=int(os.environ.get("GUILD_ID"))))
            self.synced = True
        print(f"We have logged in as {self.user}.")


client = aclient()
tree = app_commands.CommandTree(client)


def get_next_move(prompt):
    generated = tokenizer("<|startoftext|>" + " ".join(prompt), return_tensors="pt").input_ids
    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        max_length=300,
        top_p=0.95,
        temperature=1.9,
        num_return_sequences=5,
    )

    legal_move_count = defaultdict(int)
    for sample_output in sample_outputs:
        move = tokenizer.decode(sample_output, skip_special_tokens=True)[len(" ".join(prompt)) + 1 :]
        legal_move_count[move[: move.index(" ")]] += 1

    return sorted(legal_move_count.items(), key=lambda x: -x[1])


def display_board(USER_CHESS):
    with io.BytesIO() as binary:
        board = Generator.generate(USER_CHESS).resize((500, 500), Image.ANTIALIAS)
        board.save(binary, "PNG")
        binary.seek(0)
        file = discord.File(fp=binary, filename="board.png")

    return file


async def human_move(interaction, move):
    user_chess[interaction.user.id].push_san(move)
    user_prompt[interaction.user.id].append(move)

    embed = discord.Embed(
        title="Your game with Chess Bot",
        description="`" + str(interaction.user.name) + "` have moved: **" + move + "**",
        color=discord.Color.green(),
    )
    return embed, display_board(user_chess[interaction.user.id])


async def bot_move(interaction, embed, file):
    await interaction.response.defer(ephemeral=True)
    moves = get_next_move(user_prompt[interaction.user.id])
    await interaction.followup.send(embed=embed, file=file, ephemeral=True)
    move_success = 0
    legal_move = str(user_chess[interaction.user.id].legal_moves)
    legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

    for move in moves:
        if move[0] in legal_move:
            user_chess[interaction.user.id].push_san(move[0])
            user_prompt[interaction.user.id].append(move[0])
            move_success = 1
            break

    if not move_success:
        del user_chess[interaction.user.id]
        del user_count[interaction.user.id]
        del user_prompt[interaction.user.id]
        await interaction.followup.send("<@" + str(interaction.user.id) + "> won the game.", ephemeral=True)
        return

    legal_move = str(user_chess[interaction.user.id].legal_moves)
    legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")
    if not legal_move:
        del user_chess[interaction.user.id]
        del user_count[interaction.user.id]
        del user_prompt[interaction.user.id]
        await interaction.followup.send("<@" + str(interaction.user.id) + "> lost the game.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Your game with Chess Bot",
        description="Your opponent has moved: **" + move[0] + "**",
        color=discord.Color.green(),
    )
    embed.set_footer(text="Legal Move: " + ", ".join(legal_move))
    button = Button(label="Resign", style=discord.ButtonStyle.gray, emoji="🏳️")

    async def button_callback(interaction):
        del user_chess[interaction.user.id]
        del user_count[interaction.user.id]
        del user_prompt[interaction.user.id]
        await interaction.response.send_message(
            "<@" + str(interaction.user.id) + "> resigned the game.", ephemeral=True
        )

    button.callback = button_callback

    view = View()
    view.add_item(button)
    await interaction.followup.send(
        embed=embed, file=display_board(user_chess[interaction.user.id]), view=view, ephemeral=True
    )


user_chess = {}
user_color = {}
user_count = defaultdict(int)
user_prompt = defaultdict(list)


@tree.command(
    name="start", description="Start a playing chess game.", guild=discord.Object(id=int(os.environ.get("GUILD_ID")))
)
@app_commands.choices(
    color=[app_commands.Choice(name="White", value="White"), app_commands.Choice(name="Black", value="Black")]
)
async def start(interaction: discord.Interaction, color: app_commands.Choice[str]):

    if color.value == "White":
        user_chess[interaction.user.id] = chess.Board()
        user_color[interaction.user.id] = "White"
        legal_move = str(user_chess[interaction.user.id].legal_moves)
        legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

        embed = discord.Embed(
            title="Game Started!",
            description="Black: Chess Bot\nWhite: `"
            + str(interaction.user.name)
            + "`\n\nUse `/move` to move your piece.",
            color=discord.Color.blue(),
        )
        embed.set_footer(text="Legal Move: " + ", ".join(legal_move))
        await interaction.response.send_message(
            embed=embed, file=display_board(user_chess[interaction.user.id]), ephemeral=True
        )

    elif color.value == "Black":
        user_color[interaction.user.id] = "Black"
        user_chess[interaction.user.id] = chess.Board()
        legal_move = str(user_chess[interaction.user.id].legal_moves)
        legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")

        embed = discord.Embed(
            title="Game Started!",
            description="Black: `"
            + str(interaction.user.name)
            + "`\nWhite: Chess Bot\n\nUse `/move` to move your piece.",
            color=discord.Color.blue(),
        )
        file = display_board(user_chess[interaction.user.id])

        user_count[interaction.user.id] += 1
        user_prompt[interaction.user.id].append(str(user_count[interaction.user.id]) + ".")
        await bot_move(interaction, embed, file)


@tree.command(
    name="move",
    description="Move your piece among legal moves.",
    guild=discord.Object(id=int(os.environ.get("GUILD_ID"))),
)
async def move(interaction: discord.Interaction, move: str):
    if interaction.user.id not in user_chess.keys():
        embed = discord.Embed(
            title="You do not have a game in progress.",
            description="Use `/start` to start a game.",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    legal_move = str(user_chess[interaction.user.id].legal_moves)
    legal_move = legal_move[legal_move.index("(") + 1 : legal_move.index(")")].split(", ")
    if move not in legal_move:
        embed = discord.Embed(
            title="Illegal Move Played.", description="Legal Move: " + ", ".join(legal_move), color=discord.Color.red()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    if user_color[interaction.user.id] == "White":
        user_count[interaction.user.id] += 1
        user_prompt[interaction.user.id].append(str(user_count[interaction.user.id]) + ". ")
        embed, file = await human_move(interaction, move)
        await bot_move(interaction, embed, file)

    elif user_color[interaction.user.id] == "Black":
        embed, file = await human_move(interaction, move)
        user_count[interaction.user.id] += 1
        user_prompt[interaction.user.id].append(str(user_count[interaction.user.id]) + ".")
        await bot_move(interaction, embed, file)


client.run(os.environ.get("TOKEN"))
