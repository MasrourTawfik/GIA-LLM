Dataset format for FineTuning
========================================================

Before pushing your dataset, you need to make sure that it is in the right format. The format that we will be using is called the **Alpaca format**. The dataset should be a json file containing the following columns:

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
