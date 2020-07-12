import os

import tensorflow as tf

import time

import subprocess

# used to measure time elapsed

start = time.time()

# name of folder containing bal_train, eval, and unbal_train folders

wd = 'audioset_v1_embeddings'

# wd = 'test_tfrecord'

# class labels can be found here: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

class_label = 47

# list of all .tfrecord filenames

# filenames = [wd + '/' + x for x in os.listdir(wd)]

filenames = []

for folder in os.listdir(wd):
    for file in os.listdir(os.path.join(wd,folder)):
        
        filenames.append(os.path.join(wd,folder,file))

# create a TFRecordDataset object containing concatenated
# .tfrecord files to iterate through it

dataset = tf.data.TFRecordDataset(filenames)

"""

declare the type of context features that are in the .tfrecord files.
The keys for these context features were obtained from the official
AudioSet website.

"""

context_features_dict = {

    # parsed as tensor

    'video_id' : tf.io.FixedLenFeature(
                        [], tf.string, default_value=''),

    # parsed as tensor

    'start_time_seconds' : tf.io.FixedLenFeature(
                            [], tf.float32, default_value=0.0),

    # parsed as tensor

    'end_time_seconds' : tf.io.FixedLenFeature(
                            [], tf.float32, default_value=0.0),

    # parsed as sparse tensor

    'labels' : tf.io.VarLenFeature(dtype=tf.int64)

}

"""

declare the type of descriptive features that are in the .tfrecord
files. The key for these descriptive features was obtained from the
official AudioSet website

"""

features_dict = {

    # parsed as tensor

    'audio_embedding' : tf.io.FixedLenSequenceFeature(
                            [],dtype=tf.string)

}

# used to name the extracted audio files

file_idx = 1

# iterate through each .tfrecord file

for record in dataset:

    """

    parse each record into the .proto format that can be found at
    the official AudioSet website. Note that the descriptive
    features are still in byte-string format at this point.

    tf.io.parse_sequence_example() won't work because it
    automatically concatenates the labels together. This will
    not allow for the individual selection of records. See
    AudioSet_test1.py for more details.

    """

    seqExample = tf.io.parse_single_sequence_example(
        record,context_features_dict,features_dict)
    
    # extract the list of labels for the sound clip

    labels = seqExample[0]['labels'].values.numpy().tolist()
    
    # if the desired class label is not in this list, skip to the next
    # iteration in the for loop
    
    if class_label not in labels:
        
        continue

    # extract the YouTube video ID associated with the record

    video_id = seqExample[0]['video_id'].numpy().decode('utf-8')
    
    """
    
    YouTube uses the Opus audio codec. The following cmd.exe command returns
    all audio and video urls associated with a YouTube video. This command
    includes the option -f bestaudio[ext-webm] to choose the opus audio url
    with the highest bitrate. A list of the available YouTube itag codes can
    be found here:
        
    https://gist.github.com/sidneys/7095afe4da4ae58694d128b1034e01e2
    
    NOTE: the double quotation marks that enclose the YouTube url are
    important.
    
    """
    
    ydl_cmd = ('youtube-dl -g -f bestaudio[ext=m4a] '+
               '"https://www.youtube.com/watch?v='+video_id+'"') 
    
    # if the YouTube video is no longer available or has been taken down by its
    # author, skip to the next iteration of the for loop
    
    try:
        
        """
        
        execute in cmd.exe and return result as a byte-string. The resulting
        string can be searched for its itag code using ctrl+f. This is done by
        searching: "itag=ITAG_CODE" (without quotation marks). For example,
        searching itag=251 will check if the resulting audio url is for the
        opus audio file with a sampling rate of 48 kHz and a bitrate of
        160 kbps.
        
        The following cmd.exe command:
            
        youtube-dl -F "YOUTUBE_URL"
        
        lists all the audio and video formats available for download with their
        associated itag codes. For example:
            
        youtube-dl -F "https://www.youtube.com/watch?v=x_R-qzjZrKQ"
        
        """
        
        audio_url = subprocess.check_output(ydl_cmd)
        
    except subprocess.CalledProcessError:
        
        continue
    
    # decode resulting byte string to string
    
    audio_url = audio_url.decode('utf-8')
    
    # remove the last newline character
    
    audio_url = audio_url[:-1]
    
    # extract the start time in seconds of the sound clip

    start_time_sec = seqExample[0]['start_time_seconds'].numpy()

    # extract the end time in seconds of the sound clip

    end_time_sec = seqExample[0]['end_time_seconds'].numpy()

    # start time string for the sound clip. This is in the format
    # 'HH:MM:SS'

    start_time = time.strftime("%H:%M:%S",time.gmtime(start_time_sec))

    # increment string for the sound clip. This is in the format
    # 'HH:MM:SS'. Since almost all YouTube videos are at least 10 seconds long,
    # then this will be '00:00:10' for most videos. However, in the case of
    # videos that are less than 10 seconds long, this will be different.
    
    increment = time.strftime("%H:%M:%S",time.gmtime(
                              end_time_sec - start_time_sec))
    
    """
    
    create the appropriate ffmpeg command and execute it in cmd.exe.
    
    NOTE: the double quotation marks that enclose the audio_url are important.
    See:
        
    https://unix.stackexchange.com/questions/427891/ffmpeg-youtube-dl
    
    for details. This command won't work otherwise.
    
    """
    
    ffmpeg_cmd = ('ffmpeg -ss '+start_time+
                  ' -i "'+audio_url+
                  '" -t '+increment+
                  ' -c copy cropped_audio/'+str(file_idx)+'.m4a')
    
    try:
    
        # execute command in cmd.exe using ffmpeg.exe
    
        subprocess.check_output(ffmpeg_cmd)
    
    except subprocess.CalledProcessError:
        
        continue
    
    # convert to a .wav file
    
    subprocess.call('ffmpeg -i cropped_audio/'+str(file_idx)+'.m4a cropped_audio/'+str(file_idx)+'.wav')
    
    # delete the .m4a file as it is no longer needed
    
    os.remove('cropped_audio/'+str(file_idx)+'.m4a')
    
    # increment the index used to name the audio files
    
    file_idx = file_idx + 1

# close ffmpeg.exe once done

subprocess.call('taskkill /IM ffmpeg.exe')

end = time.time()

print('Time elapsed: {}'.format(time.strftime(
                                "%H:%M:%S",time.gmtime(end-start))))