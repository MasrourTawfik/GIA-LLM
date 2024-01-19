Workflow Diagram Explanation
=============================

This diagram illustrates the process of enhancing Large Language Models (LLMs) with custom adapters and integrating them into a user-friendly interface for both end-users and developers.

.. figure:: /Documentation/images/llms2.JPG
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: pipeline

   The project's pipeline.

Benchmarking
--------------------------------

The workflow begins with the benchmarking phase, where the performance of various open-source LLMs is compared. The most suitable models are selected as baseline models for further development.

.. figure:: /Documentation/images/benchm.JPG
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: benchmarking_phase

   Benchmarking various open source LLMs.

Fine-tuning & Adapter Creation
--------------------------------

Next, the selected models undergo fine-tuning to optimize them for specific tasks. Custom adapters are then created to augment the LLMs' abilities, which are saved on the Hugging Face platform, a hub for sharing machine learning models.

.. hint::
   For more details, refer to the *FineTuning ludwig* :doc:`tutorial <3_1_FineTuning_Ludwig>`.

Models with Adapters
--------------------------------

These custom adapters are integrated into the baseline models. The integration allows for a more modular approach to machine learning model enhancement, where specific capabilities can be added without altering the entire model architecture.

.. hint::
   For more details, refer to the *Load adapter and attach to model* :doc:`tutorial <3_3_Load_adapter_and_attach_to_model>`.

Synthetic Data Generation
--------------------------------

In parallel to adapter integration, synthetic data is generated using GPT-4. This data can be used to further train and refine the models, ensuring that they are well-equipped to handle a variety of scenarios.

.. hint::
   For more details, refer to the *Synthetic data* :doc:`tutorial <4_2_Synthetic_data>`.

Interface
--------------------------------

Finally, the models with adapters are made accessible through two distinct interfaces:

* User Interface
  Designed for end-users, this interface is user-friendly and allows users to select between default models or to perform custom fine-tuning with their own data.

* Developer Interface
  Tailored for developers, this interface provides the tools needed to manage models, adapters, and test results effectively.

This diagram encapsulates the streamlined approach to adapting LLMs to specialized tasks and ensuring that both users and developers have the necessary tools at their disposal.

.. note:: 
    In the final version of the interface, the Developer Interface will not be present, as the system is designed to be user-centric without requiring direct developer involvement.