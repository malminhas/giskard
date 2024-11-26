# giskard
Various experiments with [`giskard`](https://docs.giskard.ai/) an open-source framework for testing ML models, from LLMs to tabular models.  Explorations include:

* [giskard-playbook.ipynb](giskard-playbook.ipynb): outline a recipe to build a Retrieval Augmented Generation (RAG) model that answers questions about climate change, based on the 2023 Climate Change Synthesis Report by the IPCC. `giskard` LLM Scan is used to automatically detect issues with the model by generating scan results such as the [hallucination results](https://htmlpreview.github.io/?https://github.com/malminhas/giskard/blob/main/hallucination_results.html) in the notebook.
* [climate-oracle.py](climate-oracle.py): a CLI tool built from the code in [giskard-playbook.ipynb](giskard-playbook.ipynb).  Example output on execution:

<code>
$ python climate-oracle.py
Generating create_retrieval_chain...
Enter a climate question (or press Enter to quit). eg. "Is sea level rise avoidable and when will it stop?" 
> What are the key mitigations to avoid the worse impact from climate change?
According to the IPCC Climate Change Synthesis Report (2023), the key mitigations to avoid the worst impacts of climate change include reducing greenhouse gas emissions, increasing the use of renewable energy sources, improving energy efficiency, and implementing sustainable land management practices. Additionally, the report emphasizes the importance of transitioning to a low-carbon economy and investing in adaptation measures to reduce vulnerability to climate change impacts. It also highlights the need for international cooperation and policy changes to support these mitigation efforts. Overall, a combination of actions at the individual, community, and global levels is necessary to effectively mitigate the impacts of climate change.
> Is sea level rise avoidable and when will it stop?
According to the IPCC Climate Change Synthesis Report (2023), sea level rise is a result of global warming and is expected to continue for centuries to come. While it is not possible to completely avoid sea level rise, the report states that taking immediate and ambitious actions to reduce greenhouse gas emissions can slow down the rate of sea level rise. However, even with significant emissions reductions, sea level rise will continue for centuries due to the long-term effects of past and current emissions. The report also notes that the exact timing and magnitude of sea level rise is uncertain and will depend on future emissions and the response of the Earth's climate system. Therefore, it is crucial for countries to work together to reduce emissions and adapt to the impacts of sea level rise.`
</code>
