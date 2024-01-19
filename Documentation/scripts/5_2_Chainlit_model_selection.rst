Chainlit - model selection
========================================================

In this part, we will see how we can add a dropdown menu to give the user the ability to choose the model he wants to use.

.. note:: 
    Make sure that the ``chainlit`` package is installed. If not, run ``pip install chainlit``.

    There is a video tutorial available for this section `watch it <https://drive.google.com/file/d/1yhFJByI0qRtNXkf_Ndygyuts8MZk4aH_/view?usp=drive_link>`_.

The implementation
------------------

Let's first import the necessary libraries:

.. code-block:: python

    import chainlit as cl
    from chainlit.input_widget import Select

To create a dropdown menu, we will be using the ``Select`` widget. This widget takes a list of options as input and returns the selected option. Let's see how to do this.

.. code-block:: python

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Choos an open source llm",
                values=["mistral-7b", "llama2-7b", "zephyr-7b-beta"], # the list of models goes here
                initial_index=0,
            ),
        ]
    ).send()

To verify if the selected model is being used we will run the following program that will send a message with the selected model to the user.

.. code-block:: python

    import chainlit as cl
    from chainlit.input_widget import Select

    @cl.on_chat_start
    async def start():
        settings = await cl.ChatSettings(
            [
                Select(
                    id="model",
                    label="Choos an open source llm",
                    values=["mistral-7b", "llama2-7b", "zephyr-7b-beta"],
                    initial_index=0,
                ),
            ]
        ).send()


    @cl.on_settings_update
    async def setup_agent(settings):
        model = settings["model"]
        await main(cl.Message(content=f"Running model: {model}"))


    @cl.on_message
    async def main(message: cl.Message):
        await message.send()

.. figure:: /Documentation/images/model_dropdown_selection.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: The dropdown menu.

   The dropdown menu.

After clicking on the ``confirm`` button. The name of the selected model will be printed to the user.

.. figure:: /Documentation/images/model_dropdown_name_printed.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: The selected model printed.

   The name of the selected model is printed in the UI.
