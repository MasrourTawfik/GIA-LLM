Host the dataset in HuggingFace
=======================================

We have seen in the previous tutorial :doc:`FineTuning_Ludwig <3_1_FineTuning_Ludwig>` that in order to fine-tune a model on a new task, we need to have a dataset in the right format. In this tutorial, we will see how to push your dataset to HuggingFace so that it can be used later to fine-tune a model.

.. note:: 
    You need to create an account on HuggingFace to be able to push your dataset. You can create an account `here <https://huggingface.co/>`_.

    There is a video tutorial available for this section `watch it <put_the_link_here>`_.


Pushing the dataset
--------------------

Once your dataset is in the right format (see the data format :doc:`tutorial <4_1_DataFormat>`), you can upload it to HuggingFace. To do so, you need to follow the following steps:

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
