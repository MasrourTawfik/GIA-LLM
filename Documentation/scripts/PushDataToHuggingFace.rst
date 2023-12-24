How to push your dataset to HuggingFace
========================================================

We have seen in the previous tutorial `Fine-tuning_using_Ludwig` that in order to fine-tune a model on a new task, we need to have a dataset in the right format. In this tutorial, we will see how to push your dataset to HuggingFace so that it can be used later to fine-tune a model.

.. note:: 
    You need to create an account on HuggingFace to be able to push your dataset. You can create an account `here <https://huggingface.co/>`_.

    There is a video tutorial available for this section `here <put_the_link_here>`_.


Data format
--------

Before pushing your dataset, you need to make sure that it is in the right format. The dataset should be a json file containing the following columns:

- instruction : the instruction to execute
- input : the input of the instruction
- output : the output of the instruction

Here is an example of a dataset in the right format:

.. code-block:: json

    [
        {
            "instruction": "Assess the company's waste management practices.",
            "input": "How does your company manage hazardous and non-hazardous waste?",
            "output": "We segregate hazardous and non-hazardous waste at the source. Hazardous waste is handled by a licensed contractor, and we regularly audit their practices for compliance. Non-hazardous waste is recycled where possible."
        },
        {
            "instruction": "Evaluate the company's water usage efficiency.",
            "input": "What measures has your company implemented to reduce water consumption?",
            "output": "We have installed low-flow fixtures and automated sensors in our facilities. We also use a water recycling system in our manufacturing process to reduce overall water consumption."
        },
        {
            "instruction": "Check the company's energy efficiency initiatives.",
            "input": "What steps has your company taken to improve energy efficiency?",
            "output": "We've upgraded to energy-efficient lighting, HVAC systems, and machinery. Additionally, we conduct regular energy audits to identify further improvement areas."
        },
        {
            "instruction": "Review the company's air emissions control.",
            "input": "How does your company manage and monitor air emissions?",
            "output": "We have installed emission control systems like scrubbers and electrostatic precipitators. We conduct regular emissions testing and maintain records as per regulatory requirements."
        }
    ]

Pushing your dataset
--------------------

Once your dataset is in the right format, you can upload it to HuggingFace. To do so, you need to follow the following steps:

1. Connect to your HuggingFace account.
2. Visit the following `link <https://huggingface.co/new-dataset>`_ and give a name to your dataset, choose a proper license and choose if you want to make your dataset public or private.
   
.. figure:: /Documentation/images/Push_data_1.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: step_1_push_data

   Create a new dataset page.

3. Click on the `Files and versions` tab.

.. figure:: /Documentation/images/Push_data_2.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: step_2_push_data

   The `Files and versions` tab.

4. Click on `Add file`, then `Upload files` and upload your dataset.

.. figure:: /Documentation/images/Push_data_3.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: step_3_push_data

   The `Files and versions` tab.

5. Drag and drop your dataset.

.. figure:: /Documentation/images/Push_data_4.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: step_4_push_data

   The drag and drop area.

6. Hit `Commit changes to main`.

.. figure:: /Documentation/images/Push_data_5.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: step_5_push_data

   The `Commit changes to main` button.

Congratulations! You have successfully pushed your dataset to HuggingFace. You can now use it to fine-tune a model on a new task.

.. figure:: /Documentation/images/Push_data_6.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: step_6_push_data

   The preview of the data.
