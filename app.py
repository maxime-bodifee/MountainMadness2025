from shiny import App, ui, reactive

app_ui = ui.page_fillable(
    ui.panel_title("Prevent the Descent"),
    ui.chat_ui("chat"),
    fillable_mobile=True,
)

# Create a welcome message
welcome = """
Welcome to the Suicide Prevention Software. Please lookout for any alerts below.
"""


@reactive.file_reader("./alert.txt")
def alert() -> str:
    with open("./alert.txt", encoding="utf-8") as file:
        return file.read()


def server(input, output, session):
    chat = ui.Chat(id="chat", messages=[welcome])

    @reactive.effect
    async def handle_new_alert():
        new_alert = alert()
        if new_alert:
            await chat.append_message(new_alert)


app = App(app_ui, server)


if __name__ == '__main__':
    app.run()
