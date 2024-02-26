# import datetime
# import os
#
# import requests
# import re
# import time
# import csv
# import json
# import random
# import pprint
#
# headers = '''User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'''
#
#
# def get_headers(header_raw):
#     return dict(line.split(": ", 1) for line in header_raw.split("\n"))
#
#
# # 获取网页源码的文本文件
# def get_html(url):
#     response = requests.get(url, headers=get_headers(headers), timeout=20)
#     response.close()
#     return response
#
#
# def getMD(url, name):
#     res = get_html(url).text
#     content = re.findall('<div itemprop="articleBody">(.*?)<footer>', res, re.S)[0]
#     content = content.split('</style>')[-1]
#     content = re.sub('<span class="eqno">.*?</span>', '', content)
#     content = re.sub('<div .*?>', '', content)
#     content = re.sub('\[\d+\]:', '', content)
#     content = content.replace('<pre>', '\n```python\n').replace('</pre>', '\n```\n').replace('</code>', '')
#     content = content.replace('<strong>', '**').replace('</strong>', '**')
#     content = re.sub(r'<span class="pre">([^<]*)</span>', r'`\1`', content)
#     content = re.sub('<span .*?>', '', content)
#     content = re.sub('<ul.*?>', '', content)
#     content = re.sub('</ul>', '\n', content)
#     content = re.sub('<li.*?>', '\n- ', content)
#     content = re.sub('<script.*?</script>', '', content)
#     content = re.sub('<iframe .*?>', '', content)
#     content = re.sub('<span>', '', content)
#     content = re.sub('</span>', '', content)
#     content = re.sub('<dd.*?>', '', content)
#     content = re.sub('<dl.*?>', '', content)
#     content = re.sub('<dt.*?>', '', content)
#     content = re.sub('<code .*?>', '', content)
#     content = re.sub('<p class="admonition-title">', '\n### ', content)
#     content = content.replace('</div>', '').replace('&amp;', '&').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', '\'').replace('&#64;', '@')
#     content = content.strip()
#
#     content = content.replace('<h1>', "\n# ").replace('<h2>', '\n## ').replace('<h3>', '\n### ').replace('<h4>', '\n#### ').replace('<h5>', '\n##### ').replace('<h6>', '\n###### ')
#     content = content.replace('</h1>', '').replace('</h2>', '\n').replace('</h3>', '\n').replace('</h4>', '\n').replace('</h5>', '').replace('</h6>', '')
#     content = content.replace('<p>', '\n\n').replace('</p>', '').replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
#     content = content.replace('</center>', '').replace('</iframe>', '').replace('<center>', '')
#     content = content.replace('\(', '$').replace('\)', '$').replace('\[', '\n$$\n').replace('\]', '\n$$\n')
#     content = content.replace('```python\n\n\n```', '').replace('\n\n', '\n').replace('href="', 'href="https://nvlabs.github.io/sionna/').replace('src="..', 'src="https://nvlabs.github.io/sionna')
#     content = re.sub('\n\n\n', '\n', content)
#     content = content.replace('</li>', '').replace('</dd>', '').replace('</dl>', '').replace('</dt>', '')
#     name = name.replace(':', '')
#     with open('sionna/' + name + '.md', 'w', encoding='utf-8') as f:
#         f.write(content)
#
#
# if __name__ == '__main__':
#     if not os.path.exists('sionna'):
#         os.mkdir('sionna')
#     # infoList = [('https://nvlabs.github.io/sionna/quickstart.html', 'Quickstart'), ('https://nvlabs.github.io/sionna/installation.html', 'Installation'), ('https://nvlabs.github.io/sionna/examples/Hello_World.html', '“Hello, world!”'), ('https://nvlabs.github.io/sionna/examples/Discover_Sionna.html', 'Discover Sionna'), ('https://nvlabs.github.io/sionna/tutorials.html', 'Tutorials'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html', 'Part 2: Differentiable Communication Systems'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part3.html', 'Part 3: Advanced Link-level Simulations'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html', 'Part 4: Toward Learned Receivers'), ('https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html', 'Basic MIMO Simulations'), ('https://nvlabs.github.io/sionna/examples/Pulse_shaping_basics.html', 'Pulse-shaping Basics'), ('https://nvlabs.github.io/sionna/examples/Optical_Lumped_Amplification_Channel.html', 'Optical Channel with Lumped Amplification'), ('https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html', '5G Channel Coding and Rate-Matching: Polar vs.\xa0LDPC Codes'), ('https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html', '5G NR PUSCH Tutorial'), ('https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html', 'Bit-Interleaved Coded Modulation (BICM)'), ('https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html', 'MIMO OFDM Transmissions over the CDL Channel Model'), ('https://nvlabs.github.io/sionna/examples/Neural_Receiver.html', 'Neural Receiver for OFDM SIMO Systems'), ('https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html', 'Realistic Multiuser MIMO OFDM Simulations'), ('https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html', 'OFDM MIMO Channel Estimation and Detection'), ('https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html', 'Introduction to Iterative Detection and Decoding'), ('https://nvlabs.github.io/sionna/examples/Autoencoder.html', 'End-to-end Learning with Autoencoders'), ('https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html', 'Weighted Belief Propagation Decoding'), ('https://nvlabs.github.io/sionna/examples/CIR_Dataset.html', 'Channel Models from Datasets'), ('https://nvlabs.github.io/sionna/examples/DeepMIMO.html', 'Using the DeepMIMO Dataset with Sionna'), ('https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html', 'Introduction to Sionna RT'), ('https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html', 'Tutorial on Diffraction'), ('https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Scattering.html', 'Tutorial on Scattering'), ('https://nvlabs.github.io/sionna/made_with_sionna.html', '“Made with Sionna”'), ('https://nvlabs.github.io/sionna/em_primer.html', 'Primer on Electromagnetics'), ('https://nvlabs.github.io/sionna/api/sionna.html', 'API Documentation'), ('https://nvlabs.github.io/sionna/api/fec.html', 'Forward Error Correction (FEC)'), ('https://nvlabs.github.io/sionna/api/fec.linear.html', 'Linear Codes'), ('https://nvlabs.github.io/sionna/api/fec.ldpc.html', 'Low-Density Parity-Check (LDPC)'), ('https://nvlabs.github.io/sionna/api/fec.polar.html', 'Polar Codes'), ('https://nvlabs.github.io/sionna/api/fec.conv.html', 'Convolutional Codes'), ('https://nvlabs.github.io/sionna/api/fec.turbo.html', 'Turbo Codes'), ('https://nvlabs.github.io/sionna/api/fec.crc.html', 'Cyclic Redundancy Check (CRC)'), ('https://nvlabs.github.io/sionna/api/fec.interleaving.html', 'Interleaving'), ('https://nvlabs.github.io/sionna/api/fec.scrambling.html', 'Scrambling'), ('https://nvlabs.github.io/sionna/api/fec.utils.html', 'Utility Functions'), ('https://nvlabs.github.io/sionna/api/mapping.html', 'Mapping'), ('https://nvlabs.github.io/sionna/api/channel.html', 'Channel'), ('https://nvlabs.github.io/sionna/api/channel.wireless.html', 'Wireless'), ('https://nvlabs.github.io/sionna/api/channel.optical.html', 'Optical'), ('https://nvlabs.github.io/sionna/api/channel.discrete.html', 'Discrete'), ('https://nvlabs.github.io/sionna/api/ofdm.html', 'Orthogonal Frequency-Division Multiplexing (OFDM)'), ('https://nvlabs.github.io/sionna/api/mimo.html', 'Multiple-Input Multiple-Output (MIMO)'), ('https://nvlabs.github.io/sionna/api/nr.html', '5G NR'), ('https://nvlabs.github.io/sionna/api/rt.html', 'Ray Tracing'), ('https://nvlabs.github.io/sionna/api/signal.html', 'Signal'), ('https://nvlabs.github.io/sionna/api/utils.html', 'Utility Functions utils'), ('https://nvlabs.github.io/sionna/api/config.html', 'Configuration')]
#     infoList = [('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html', 'tutorial_for_beginners')]
#     for url, name in infoList:
#         print(url, name)
#         getMD(url, name)
# -*- coding:utf-8 -*-
# @Author: pioneer
# @Environment: Python 3.9
import datetime
import os

import requests
import re
import time
import csv
import json
import random
import pprint

headers = '''User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'''


def get_headers(header_raw):
    return dict(line.split(": ", 1) for line in header_raw.split("\n"))


# 获取网页源码的文本文件
def get_html(url):
    response = requests.get(url, headers=get_headers(headers), timeout=20)
    response.close()
    return response


def getMD(url, name):
    res = get_html(url).text
    content = re.findall('<div itemprop="articleBody">(.*?)<footer>', res, re.S)[0]
    content = content.split('</style>')[-1]
    content = re.sub('<span class="eqno">.*?</span>', '', content)
    content = re.sub('<div .*?>', '', content)
    # content = re.sub('\[\d+\]:', '', content)
    content = content.replace('<pre>', '\n```python\n').replace('</pre>', '\n```\n').replace('</code>', '')
    content = content.replace('<strong>', '**').replace('</strong>', '**')
    content = re.sub(r'<span class="pre">([^<]*)</span>', r'`\1`', content)
    content = re.sub('<span .*?>', '', content)
    content = re.sub('<ul.*?>', ' ', content)
    content = re.sub('</ul>', '\n', content)
    content = re.sub('<li.*?>', '\n- ', content)
    content = re.sub('<script.*?</script>', '', content)
    content = re.sub('<iframe .*?>', '', content)
    content = re.sub('<span>', '', content)
    content = re.sub('</span>', '', content)
    content = re.sub('<dd.*?>', '', content)
    content = re.sub('<dl.*?>', '', content)
    content = re.sub('<dt.*?>', '', content)
    content = re.sub('<code .*?>', '', content)
    content = re.sub('<p class="admonition-title">', '\n### ', content)
    content = content.replace('</div>', '').replace('&amp;', '&').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', '\'').replace('&#64;', '@')
    content = content.strip()

    content = content.replace('<h1>', "\n# ").replace('<h2>', '\n## ').replace('<h3>', '\n### ').replace('<h4>', '\n#### ').replace('<h5>', '\n##### ').replace('<h6>', '\n###### ')
    content = content.replace('</h1>', '').replace('</h2>', '\n').replace('</h3>', '\n').replace('</h4>', '\n').replace('</h5>', '').replace('</h6>', '')
    content = content.replace('<p>', '    \n\n').replace('</p>', '').replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
    content = content.replace('</center>', '').replace('</iframe>', '').replace('<center>', '')
    content = content.replace('\(', '$').replace('\)', '$').replace('\[', '\n$$\n').replace('\]', '\n$$\n')
    content = content.replace('```python\n\n\n```', '').replace('\n\n', '\n').replace('href="', 'href="https://nvlabs.github.io/sionna/').replace('src="..', 'src="https://nvlabs.github.io/sionna')
    content = re.sub('\n\n\n', '\n', content)
    content = content.replace('</li>', '').replace('</dd>', '').replace('</dl>', '').replace('</dt>', '')
    name = name.replace(':', '')
    with open('sionna/' + name + '.md', 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == '__main__':
    if not os.path.exists('../../sionna'):
        os.mkdir('../../sionna')
    # infoList = [('https://nvlabs.github.io/sionna/quickstart.html', 'Quickstart'), ('https://nvlabs.github.io/sionna/installation.html', 'Installation'), ('https://nvlabs.github.io/sionna/examples/Hello_World.html', '“Hello, world!”'), ('https://nvlabs.github.io/sionna/examples/Discover_Sionna.html', 'Discover Sionna'), ('https://nvlabs.github.io/sionna/tutorials.html', 'Tutorials'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part2.html', 'Part 2: Differentiable Communication Systems'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part3.html', 'Part 3: Advanced Link-level Simulations'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part4.html', 'Part 4: Toward Learned Receivers'), ('https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html', 'Basic MIMO Simulations'), ('https://nvlabs.github.io/sionna/examples/Pulse_shaping_basics.html', 'Pulse-shaping Basics'), ('https://nvlabs.github.io/sionna/examples/Optical_Lumped_Amplification_Channel.html', 'Optical Channel with Lumped Amplification'), ('https://nvlabs.github.io/sionna/examples/5G_Channel_Coding_Polar_vs_LDPC_Codes.html', '5G Channel Coding and Rate-Matching: Polar vs.\xa0LDPC Codes'), ('https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html', '5G NR PUSCH Tutorial'), ('https://nvlabs.github.io/sionna/examples/Bit_Interleaved_Coded_Modulation.html', 'Bit-Interleaved Coded Modulation (BICM)'), ('https://nvlabs.github.io/sionna/examples/MIMO_OFDM_Transmissions_over_CDL.html', 'MIMO OFDM Transmissions over the CDL Channel Model'), ('https://nvlabs.github.io/sionna/examples/Neural_Receiver.html', 'Neural Receiver for OFDM SIMO Systems'), ('https://nvlabs.github.io/sionna/examples/Realistic_Multiuser_MIMO_Simulations.html', 'Realistic Multiuser MIMO OFDM Simulations'), ('https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html', 'OFDM MIMO Channel Estimation and Detection'), ('https://nvlabs.github.io/sionna/examples/Introduction_to_Iterative_Detection_and_Decoding.html', 'Introduction to Iterative Detection and Decoding'), ('https://nvlabs.github.io/sionna/examples/Autoencoder.html', 'End-to-end Learning with Autoencoders'), ('https://nvlabs.github.io/sionna/examples/Weighted_BP_Algorithm.html', 'Weighted Belief Propagation Decoding'), ('https://nvlabs.github.io/sionna/examples/CIR_Dataset.html', 'Channel Models from Datasets'), ('https://nvlabs.github.io/sionna/examples/DeepMIMO.html', 'Using the DeepMIMO Dataset with Sionna'), ('https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Introduction.html', 'Introduction to Sionna RT'), ('https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Diffraction.html', 'Tutorial on Diffraction'), ('https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_Scattering.html', 'Tutorial on Scattering'), ('https://nvlabs.github.io/sionna/made_with_sionna.html', '“Made with Sionna”'), ('https://nvlabs.github.io/sionna/em_primer.html', 'Primer on Electromagnetics'), ('https://nvlabs.github.io/sionna/api/sionna.html', 'API Documentation'), ('https://nvlabs.github.io/sionna/api/fec.html', 'Forward Error Correction (FEC)'), ('https://nvlabs.github.io/sionna/api/fec.linear.html', 'Linear Codes'), ('https://nvlabs.github.io/sionna/api/fec.ldpc.html', 'Low-Density Parity-Check (LDPC)'), ('https://nvlabs.github.io/sionna/api/fec.polar.html', 'Polar Codes'), ('https://nvlabs.github.io/sionna/api/fec.conv.html', 'Convolutional Codes'), ('https://nvlabs.github.io/sionna/api/fec.turbo.html', 'Turbo Codes'), ('https://nvlabs.github.io/sionna/api/fec.crc.html', 'Cyclic Redundancy Check (CRC)'), ('https://nvlabs.github.io/sionna/api/fec.interleaving.html', 'Interleaving'), ('https://nvlabs.github.io/sionna/api/fec.scrambling.html', 'Scrambling'), ('https://nvlabs.github.io/sionna/api/fec.utils.html', 'Utility Functions'), ('https://nvlabs.github.io/sionna/api/mapping.html', 'Mapping'), ('https://nvlabs.github.io/sionna/api/channel.html', 'Channel'), ('https://nvlabs.github.io/sionna/api/channel.wireless.html', 'Wireless'), ('https://nvlabs.github.io/sionna/api/channel.optical.html', 'Optical'), ('https://nvlabs.github.io/sionna/api/channel.discrete.html', 'Discrete'), ('https://nvlabs.github.io/sionna/api/ofdm.html', 'Orthogonal Frequency-Division Multiplexing (OFDM)'), ('https://nvlabs.github.io/sionna/api/mimo.html', 'Multiple-Input Multiple-Output (MIMO)'), ('https://nvlabs.github.io/sionna/api/nr.html', '5G NR'), ('https://nvlabs.github.io/sionna/api/rt.html', 'Ray Tracing'), ('https://nvlabs.github.io/sionna/api/signal.html', 'Signal'), ('https://nvlabs.github.io/sionna/api/utils.html', 'Utility Functions utils'), ('https://nvlabs.github.io/sionna/api/config.html', 'Configuration'), ('https://nvlabs.github.io/sionna/examples/Sionna_tutorial_part1.html', 'Part 1: Getting Started with Sionna')]
    infoList = [('https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html', 'tutorials_for_experts_OFDM MIMO Channel Estimation and Detection')]
    for url, name in infoList:
        print(url, name)
        getMD(url, name)
