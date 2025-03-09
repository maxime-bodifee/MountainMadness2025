from shiny import App, ui, reactive

app_ui = ui.page_fillable(
    ui.panel_title("Prevent the Descent"),
    ui.chat_ui("chat"),
    fillable_mobile=True,
)

# Create a welcome message
welcome = """
Hi! This is a simple Shiny `Chat` UI. Enter a message below and I will
simply repeat it back to you. For more examples, see this
[folder of examples](https://github.com/posit-dev/py-shiny/tree/main/shiny/templates/chat).
"""


@reactive.file_reader("./alert.txt")
def alert() -> str:
    with open("./alert.txt", encoding="utf-8") as file:
        return file.read()


def server(input, output, session):
    chat = ui.Chat(id="chat", messages=[welcome])

    # Define a callback to run when the user submits a message
    @chat.on_user_submit
    async def handle_user_input(user_input: str):
        # Append a response to the chat
        await chat.append_message(f"You said: {user_input}")


    @reactive.effect
    async def handle_new_alert():
        new_alert = alert()
        if new_alert:
            await chat.append_message(new_alert)


app = App(app_ui, server)


if __name__ == '__main__':
    app.run()
