# The ethos of this training regime is to take what I know about the data
# and create a biased way of preprocessing in order to make loud, cyclic
# audio very recognizable. 

# I need to look at recordings of other cyclic things such as saws, drills,
# mills, etc. to see if they differ a lot between each other

# I also need to look at normal environmental sounds as well as common
# noise from our microphone. 

# Fine scale features may not be necessary at all so doing a translation subtraction
# of amplitude from an FFT might be beneficial to get rid of noise

# Feature ideas:
# - count of frequencies that have amplitudes higher than a threshold
# - bound frequencies that we care about (e.g. get rid of 0-1kHz which is common in human speech)
# 

# The average maximum amplitude in environment noise: 0.003119
# '      '       '         '     ' angle grinder    : 0.0176
# They're on a different order of magnitude 
# try: y_grinder - mean_max_y_env & y_env - mean_max_y_env