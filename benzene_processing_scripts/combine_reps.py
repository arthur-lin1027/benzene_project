"""
Helper function to concatenate a bunch of representations with the same metadata 
into the same representation
"""
#%%
import metatensor
import numpy as np

def combine_blocks(blocks: list[metatensor.TensorBlock]):
    if len(blocks) == 1:
        return blocks
    # Check all labels are unified.
    sample_labels = blocks[0].samples.names
    property_labels = blocks[0].properties.names
    component_labels = blocks[0].components.names
    for block in blocks[1:]:
        assert sample_labels == block.samples.names
        assert property_labels == block.properties.names
        assert component_labels == block.components.names
    samples = []
    properties = []
    components = []
    values = []
    for block in blocks:
        samples.append(block.samples.values)
        properties.append(block.properties.values)
        components.append(block.components.values)
        values.append(block.values)
    
    return metatensor.TensorBlock(
        values=np.vstack(values),
        samples=metatensor.Labels(
            sample_labels, np.vstack(samples)
        ),
        components=[metatensor.Labels(
            component_labels, np.vstack(components)
        )],
        properties=metatensor.Labels(
            property_labels, np.vstack(properties)
        )
    )

#%%
rep = metatensor.load("representations-2025/rep_anisoap_(5,3,7.000,GroupCH).mts")
# %%
rep
# %%
