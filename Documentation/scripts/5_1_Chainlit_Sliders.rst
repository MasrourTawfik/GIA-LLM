Chainlit - hyperparameters tuning
========================================================

To control the behavior of the llm, we can change some hyperparameters like the ``max_new_tokens``, the ``temperature`` of the sampling, or the ``top_k`` and ``top_p`` values.

.. note:: 
    For more information on the hyperparameters that you can play with, please refer to the CTransformers `documentation <https://github.com/marella/ctransformers#config>`_.

    There is a video tutorial available for this section `watch it <https://drive.google.com/file/d/1HNQcfWgIopzxnMTqv2j2zNgJHlOSzSXA/view?usp=drive_link>`_.


The implementation
------------------

Let's first import the necessary libraries:

.. code-block:: python

    import chainlit as cl
    from chainlit.input_widget import Slider, Switch

To change the hyperparameters, we will use the ``Slider`` and ``Switch`` widgets. The ``Slider`` widget allows us to change the value of a hyperparameter by moving a slider. The ``Switch`` widget allows us to change the value of a hyperparameter by switching between two values.

.. code-block:: python

    settings = await cl.ChatSettings(
        [
            Slider(
                id="Temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="Repetition Penalty",
                label="Repetition Penalty",
                initial=0.3,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="Top P",
                label="Top P",
                initial=0.7,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="Top K",
                label="Top K",
                initial=42,
                min=0,
                max=100,
                step=1,
            ),
            Slider(
                id="Max New Tokens",
                label="Max New Tokens",
                initial=256,
                min=0,
                max=1024,
                step=1,
            ),
            Switch(id="Streaming", label="Stream Tokens", initial=True),
        ]
    ).send()

.. figure:: /Documentation/images/configuration_sliders.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: The settings panel.

   The settings panel.


Now as demonstrated in the figure above the user can change the hyperparameters using the sliders. The ``Streaming`` switch allows us to stream the tokens as they are generated. If the switch is turned off, the tokens will be generated all at once.
