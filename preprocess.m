%clean_row: clean up noise in data to array of zeros
%to use: write in command window: clean_row(filename) e.g.clean_row('2_105_0311.txt')

  subject = [7 112 113 121 75 107 79 82 118 76 115 117 119 120 105 78 124]; % 124 is the virtual participant
% subject = [123 124]; % for new tcr data after covid 
plateau_threshold = 14
for subject_id = 1:2
    data = [];
    for session = 1:6
        id = subject(subject_id);
        data_one = dlmread("data_raw/"+id+"_tcr_s"+session+".txt");
        data_one = data_one(:, 2:21);
        row = zeros(1,20);
        while size(data_one,1) < 3000
            data_one = [data_one;row];
        end
        data_one = data_one(1:3000, :);
        data = [data; data_one];
    end

    mark = zeros(18000,20);
    for c=1:20
        mark = findPlateau(data,c,c,mark);
    end
    mark = max(mark')';
    counter = 0;
    for m = 1:18000
        if mark(m,1) > plateau_threshold
            data(m,:) = 0;
        end
    end
    for session_id = 1:6
        %filename = strcat("plateau_removed_data/tcr_subject_",string(subject_id),"_session_",string(session_id),".csv");
        filename = strcat("data_preprocess/tcr_plateau_removed_data/tcr_subject_",string(subject_id),"_session_",string(session_id),".csv");
        csvwrite(filename, data(1+3000*(session_id-1):3000*session_id,:));
    end
    csvwrite("new_tcr_plateau_removed_data/subject_"+subject_id+".csv", data);
end
   
function mark = findPlateau(data,c,i,mark)
    l = 1;
    val = 0;
    for r=1:18000
        if data(r,c)~=val
            len = r-l;
            for line = r-1:-1:r-len
                mark(line,i) = len;
            end
            val = data(r,c);
            l = r;
        end     
    end
    return
end

