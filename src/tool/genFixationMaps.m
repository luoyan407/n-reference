clear; 

input_f = '/path/to/input/folder';
output_f = '/path/to/output/folder';

files = dir(fullfile(input_f,'*.mat'));
files = {files.name};

for ifile = 1:numel(files)
    disp(ifile);
    filename = files{ifile};
    load(fullfile(input_f,filename));

    fixationPts = zeros(resolution);

    for i=1:size(gaze, 1)
        for j=1:size(gaze(i).fixations, 1)
            fixationPts(int64(gaze(i).fixations(j,2)), int64(gaze(i).fixations(j,1))) = 1;
        end
    end
    save(fullfile(output_f,filename), 'fixationPts', 'gaze', 'image', 'resolution');
end