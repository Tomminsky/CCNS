%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         data preprocessing                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This file was used to clean the data, convert it into ASCII code /
% dictionary indices, create the dictionary and split into low and high
% citation groups

time_whole_script=tic;
cd('/Users/Tommy/Documents/Nijmegen/Study/CCNS/final_project/')

unix('sh preprocessdata.sh');
clc
clearvars -except time_whole_script
fileID=fopen('/Users/Tommy/Documents/Nijmegen/Study/CCNS/final_project/allcrawlspreproc.txt');

data=textscan(fileID,'%s','Delimiter','\n');
fclose(fileID);

data=data{1};

datadouble=str2double(data); % convert all cells to double (not convertible cells (i.e. cells containing strings) will be marked as NaN)
citationidx=~isnan(datadouble); % check whether it was a citation
titleidx=find(citationidx)-1; % the line before always contains the title
citationidx=find(citationidx);

[~,validtitleindex]=unique(data(titleidx)); % get indices of unique titles
validcitations=datadouble(citationidx(validtitleindex)); % apply same selection to citations
validcitationsnorm=validcitations./max(validcitations); % normalization

validtitles=data(titleidx(validtitleindex)); % obtain "valid" (i.e. unique) titles

% for ASCII represented titles
titles_as_ASCII=cellfun(@double,validtitles,'unif',0); % convert to ASCII

maxtitlesize=max(cellfun(@length,titles_as_ASCII)); % get title with highest amount of characters

whitespace=32; % define decimal ASCII code for " "

% fill end of title with white space to match maximum length
new_titles_as_ASCII=zeros(size(titles_as_ASCII,1),maxtitlesize)+whitespace;

for n=1:size(titles_as_ASCII,1)
    sizeoldstring=size(titles_as_ASCII{n},2);
    new_titles_as_ASCII(n,1:sizeoldstring)=titles_as_ASCII{n};
end

low_25_percent = max(prctile(validcitations,[0 25]));
high_25_percent = min(prctile(validcitations,[75 100]));

idxlow=validcitations<=low_25_percent;
idxhigh=validcitations>=high_25_percent;

low_25_percentsel_cit=validcitations(idxlow);
high_25_percentsel_cit=validcitations(idxhigh);

low_25_percentsel_citnorm=validcitationsnorm(idxlow);
high_25_percentsel_citnorm=validcitationsnorm(idxhigh);

low_25_percentsel_titles=new_titles_as_ASCII(idxlow,:);
high_25_percentsel_titles=new_titles_as_ASCII(idxhigh,:);

mergedvaltitleesel=[[low_25_percentsel_titles;high_25_percentsel_titles],[zeros(size(low_25_percentsel_titles(:,1)));ones(size(high_25_percentsel_titles(:,1)))]];

mergedvalcitesel=[[low_25_percentsel_cit;high_25_percentsel_cit],[zeros(size(low_25_percentsel_cit(:,1)));ones(size(high_25_percentsel_cit(:,1)))]];

mergedvalcitenormsel=[[low_25_percentsel_citnorm;high_25_percentsel_citnorm],[zeros(size(low_25_percentsel_citnorm(:,1)));ones(size(high_25_percentsel_citnorm(:,1)))]];

% save data
save('titlesASCII','new_titles_as_ASCII')
titlesASCII=new_titles_as_ASCII;
titlesASCII=array2table(titlesASCII);
writetable(titlesASCII)
save('titlesASCII_low','low_25_percentsel_titles')
titlesASCII_low=low_25_percentsel_titles;
titlesASCII_low=array2table(titlesASCII_low);
writetable(titlesASCII_low)
save('titlesASCII_high','high_25_percentsel_titles')
titlesASCII_high=high_25_percentsel_titles;
titlesASCII_high=array2table(titlesASCII_high);
writetable(titlesASCII_high)
save('titlesASCII_high_low_merged','mergedvaltitleesel')
titlesASCII_high_low_merged=mergedvaltitleesel;
titlesASCII_high_low_merged=array2table(titlesASCII_high_low_merged);
writetable(titlesASCII_high_low_merged)


save('citations_raw','validcitations')
citations_raw=validcitations;
citations_raw=array2table(citations_raw);
writetable(citations_raw)
save('citations_raw_low','low_25_percentsel_cit')
citations_raw_low=low_25_percentsel_cit;
citations_raw_low=array2table(citations_raw_low);
writetable(citations_raw_low)
save('citations_raw_high','high_25_percentsel_cit')
citations_raw_high=high_25_percentsel_cit;
citations_raw_high=array2table(citations_raw_high);
writetable(citations_raw_high)
save('citations_raw_high_low_merged','mergedvalcitesel')
citations_raw_high_low_merged=mergedvalcitesel;
citations_raw_high_low_merged=array2table(citations_raw_high_low_merged);
writetable(citations_raw_high_low_merged)

save('citations_norm','validcitationsnorm')
citations_norm=validcitationsnorm;
citations_norm=array2table(citations_norm);
writetable(citations_norm)
save('citations_norm_low','low_25_percentsel_citnorm')
citations_norm_low=low_25_percentsel_citnorm;
citations_norm_low=array2table(citations_norm_low);
writetable(citations_norm_low)
save('citations_norm_high','high_25_percentsel_citnorm')
citations_norm_high=high_25_percentsel_citnorm;
citations_norm_high=array2table(citations_norm_high);
writetable(citations_norm_high)
save('citations_norm_high_low_merged','mergedvalcitenormsel')
citations_norm_high_low_merged=mergedvalcitenormsel;
citations_norm_high_low_merged=array2table(citations_norm_high_low_merged);
writetable(citations_norm_high_low_merged)

%%% for dictionary represented titles
clc
clearvars -except time_whole_script
load('titlesASCII')
load('citations_raw')
load('citations_norm')
%

max_spec_char=20;
max_words_in_dict=20000;

dictionary_titles_raw=lower(char(new_titles_as_ASCII));

% special characters to be included in the dictionary
listofallowedspecialcharacters=[cellstr(char([33:47,58:64,91:96,123:129])')',{'§','°','α','β','γ','–','…'}];

dictionary_titles_raw_SC=[char(zeros(size(dictionary_titles_raw,1),1)+32),...
    dictionary_titles_raw,...
    char(zeros(size(dictionary_titles_raw,1),190)+32)];

max_size=size(dictionary_titles_raw_SC,2)-2;
occurancespecial_char=zeros(size(listofallowedspecialcharacters));
max_add=0;
max_tmp=0;

num_special_char_in_titles=zeros(size(dictionary_titles_raw_SC,1),1);
contains_three_dots=zeros(size(dictionary_titles_raw_SC,1),1);

for n=1:size(listofallowedspecialcharacters,2)
    disp(['evaluating character ' listofallowedspecialcharacters{n}])
    newchar=listofallowedspecialcharacters{n};
    idxraw=dictionary_titles_raw_SC==newchar;
    [row,col]=find(idxraw);
    occurancespecial_char(n)=size(row,1);
    
    [row,I]=sort(row);
    col=col(I);
    
    col_SC=[col-1,col,col+1];
    
    col_counter=0;
    for k=2:size(row,1)
        if row(k)==row(k-1)
            col_counter=col_counter+2;
            col_SC(k,:)=col_SC(k,:)+col_counter;
            max_tmp=max_tmp+1;
            if max_add<max_tmp
                max_add=max_tmp;
            end
            num_special_char_in_titles(row(k))=num_special_char_in_titles(row(k))+1;
        else
            max_tmp=0;
            col_counter=0;
        end
    end
    
    [~,I]=sort(col_SC(:,2));
    col_SC=col_SC(I,:);
    row=row(I);
    
    for m=1:size(row,1)
        new_title=[dictionary_titles_raw_SC(row(m),1:col_SC(m,1)),' ',newchar,' ',dictionary_titles_raw_SC(row(m),col_SC(m,3):max_size)];
        dictionary_titles_raw_SC(row(m),1:size(new_title,2))=new_title;
        if listofallowedspecialcharacters{n}=='…'
            contains_three_dots(row(m))=1;
        end
    end
    
end
dictionary_titles_raw_SC(dictionary_titles_raw_SC==0)=32;
disp('example:')
dictionary_titles_raw_SC(200,:)

%%% exclude all having too many special characters
idx_too_many_special=num_special_char_in_titles>max_spec_char;

%%% exclude all having '…'
idx_contains_three_dots=contains_three_dots;

idx_validtitles_remaining = ~idx_too_many_special & ~idx_contains_three_dots;

dictionary_titles_SC=dictionary_titles_raw_SC(idx_validtitles_remaining,:);
new_titles_as_ASCII=new_titles_as_ASCII(idx_validtitles_remaining,:);

remaining_validcitations=validcitations(idx_validtitles_remaining,:);
remaining_validcitationsnorm=validcitationsnorm(idx_validtitles_remaining,:);

%%% exclude weird stuff from dictionary %%%
listofallowedspecialcharacters=listofallowedspecialcharacters([listofallowedspecialcharacters{:}]~='…');

all_words=arrayfun(@(x) strsplit(dictionary_titles_SC(x,:)),1:size(dictionary_titles_SC,1),'unif',0);
[dictionary,~,c]=unique([all_words{:},listofallowedspecialcharacters(:)']);
occurance = hist(c,length(dictionary));

% include: valid variable names, defined special characters, strings that
% start with a number
idx_empty_cells=cellfun(@length ,dictionary)<1;
dictionary_new=dictionary(~idx_empty_cells);
occurance_new=occurance(~idx_empty_cells);

[~,min_occurance_per_word_in_dict]=min(arrayfun(@(x) abs(max_words_in_dict-sum(occurance_new>x)),1:200));

unknowncharidx=occurance_new<min_occurance_per_word_in_dict;
unknownwords=dictionary_new(unknowncharidx);

dictionary_new(unknowncharidx)={'<unknown>'};



idx_valid_entry=(cellfun(@isvarname,dictionary_new) | ...
    ismember(dictionary_new,listofallowedspecialcharacters) | ...
    arrayfun(@(x) ~isnan(str2double(dictionary_new{x}(1))),1:size(dictionary_new,2)) ...
    ) & ...
    occurance_new>=min_occurance_per_word_in_dict;

dictionaryunique=[dictionary_new(idx_valid_entry),'<unknown>'];

%
titles_as_dictionaryindices=nan(size(new_titles_as_ASCII));
dictionary_titles=cellfun(@strsplit,cellstr(dictionary_titles_SC(:,2:end)),'un',0);
numeltitles=numel(dictionary_titles);

progress=1/numeltitles;
timeleft=nan(100,1);
actualidx=1:numel(dictionaryunique);
m=0;

dictionaryuniquethresh=sort(dictionaryunique);



% actual word index coding
idx_contains_not_in_dict_words=zeros(size(dictionary_titles,1),1);

for n=1:numeltitles
    tic
    wordidx=arrayfun(@(x) find(strcmp(dictionary_titles{n}{x},dictionaryuniquethresh)),1:size(dictionary_titles{n},2),'unif',0);
    wordidx=[wordidx{:}];
    sizewordidx=size(wordidx,2);
    titles_as_dictionaryindices(n,1:sizewordidx)=wordidx;
    titles_as_dictionaryindices(n,sizewordidx+1:end)=zeros;
    dictionary_titles{n}(~ismember(dictionary_titles{n},dictionaryuniquethresh))={'<unknown>'};
    idx_contains_not_in_dict_words(n)=sum(ismember(dictionary_titles{n},dictionaryuniquethresh)==0)>0;
    m=m+1;
    currtime=toc;
    timeleft(m)=currtime*numeltitles-currtime*n;
    if rem(n,100)==1
        m=0;
        disp(['progress :' num2str(n/numeltitles*100) '% - time left: ' num2str(nanmean(timeleft)/60) 'm'])
    end
end

% check if title contains words that are not in dictionary

titles_as_dictionaryindices=titles_as_dictionaryindices(~idx_contains_not_in_dict_words,:);
remaining_validcitations=remaining_validcitations(~idx_contains_not_in_dict_words,:);
remaining_validcitationsnorm=remaining_validcitationsnorm(~idx_contains_not_in_dict_words,:);


validcitations=remaining_validcitations;
validcitationsnorm=remaining_validcitationsnorm;
new_titles_as_ASCII=titles_as_dictionaryindices;

low_25_percent = max(prctile(validcitations,[0 25]));
high_25_percent = min(prctile(validcitations,[75 100]));

idxlow=validcitations<=low_25_percent;
idxhigh=validcitations>=high_25_percent;

low_25_percentsel_cit=validcitations(idxlow);
high_25_percentsel_cit=validcitations(idxhigh);

low_25_percentsel_citnorm=validcitationsnorm(idxlow);
high_25_percentsel_citnorm=validcitationsnorm(idxhigh);

low_25_percentsel_titles=new_titles_as_ASCII(idxlow,:);
high_25_percentsel_titles=new_titles_as_ASCII(idxhigh,:);

mergedvaltitleesel=[[low_25_percentsel_titles;high_25_percentsel_titles],[zeros(size(low_25_percentsel_titles(:,1)));ones(size(high_25_percentsel_titles(:,1)))]];

mergedvalcitesel=[[low_25_percentsel_cit;high_25_percentsel_cit],[zeros(size(low_25_percentsel_cit(:,1)));ones(size(high_25_percentsel_cit(:,1)))]];

mergedvalcitenormsel=[[low_25_percentsel_citnorm;high_25_percentsel_citnorm],[zeros(size(low_25_percentsel_citnorm(:,1)));ones(size(high_25_percentsel_citnorm(:,1)))]];

save('dictionary','dictionaryuniquethresh')
dictionary=dictionaryuniquethresh';
dictionary=cell2table(dictionary);
writetable(dictionary)

save('titlesDict','titles_as_dictionaryindices')
titlesDict=titles_as_dictionaryindices;
titlesDict=array2table(titlesDict);
writetable(titlesDict)

save('titlesDict_low','low_25_percentsel_titles')
titlesASCII_low=low_25_percentsel_titles;
titlesDict_low=array2table(titlesASCII_low);
writetable(titlesDict_low)

save('titlesDict_high','high_25_percentsel_titles')
titlesDict_high=high_25_percentsel_titles;
titlesDict_high=array2table(titlesDict_high);
writetable(titlesDict_high)

save('titlesDict_high_low_merged','mergedvaltitleesel')
titlesDict_high_low_merged=mergedvaltitleesel;
titlesDict_high_low_merged=array2table(titlesDict_high_low_merged);
writetable(titlesDict_high_low_merged)

save('citationsDict','remaining_validcitations')
citationsDict=remaining_validcitations;
citationsDict=array2table(citationsDict);
writetable(citationsDict)

save('citationsDict_low','low_25_percentsel_cit')
citations_raw_low=low_25_percentsel_cit;
citationsDict_low=array2table(citations_raw_low);
writetable(citationsDict_low)

save('citationsDict_high','high_25_percentsel_cit')
citations_raw_high=high_25_percentsel_cit;
citationsDict_high=array2table(citations_raw_high);
writetable(citationsDict_high)

save('citationsDict_high_low_merged','mergedvalcitesel')
citations_raw_high_low_merged=mergedvalcitesel;
citationsDict_high_low_merged=array2table(citations_raw_high_low_merged);
writetable(citationsDict_high_low_merged)

save('citationsnormDict','remaining_validcitationsnorm')
citationsnormDict=remaining_validcitationsnorm;
citationsnormDict=array2table(citationsnormDict);
writetable(citationsnormDict)

save('citationsnormDict_low','low_25_percentsel_cit')
citations_raw_low=low_25_percentsel_citnorm;
citationsnormDict_low=array2table(citations_raw_low);
writetable(citationsnormDict_low)

save('citationsnormDict_high','high_25_percentsel_cit')
citations_raw_high=high_25_percentsel_citnorm;
citationsnormDict_high=array2table(citations_raw_high);
writetable(citationsnormDict_high)

save('citationsnormDict_high_low_merged','mergedvalcitenormsel')
citations_raw_high_low_merged=mergedvalcitenormsel;
citationsnormDict_high_low_merged=array2table(citations_raw_high_low_merged);
writetable(citationsnormDict_high_low_merged)

% again for ASCII represented titles for making ASCII for same titles as Dict
new_ASCII=table2array(titlesDict);
dictionaryuniquethresh=[dictionaryuniquethresh,{' '}];
whitespace=find(strcmp(dictionaryuniquethresh,{' '}));
new_ASCII(new_ASCII==0)=whitespace;

max_size_title=size(new_ASCII,2);

m=0;
numeltitles=size(new_ASCII,1);
for title_=1:numeltitles
tic
[last_word,s]=bwlabel(new_ASCII(title_,1:max_size_title)==whitespace);
idx=find(last_word==s);
idx=idx(1)-1;

space_requirement_vec=1:2:2*idx-1;

title_curr=new_ASCII(title_,1:idx);

curr_space_requirement_vec=1:idx;

title_curr_spaced=[zeros(size(space_requirement_vec)),zeros(size(curr_space_requirement_vec))]+whitespace;
title_curr_spaced(space_requirement_vec)=new_ASCII(title_,curr_space_requirement_vec);

new_ASCII(title_,1:size(title_curr_spaced,2))=title_curr_spaced;
title_ASCII=double([dictionaryuniquethresh{new_ASCII(title_,:)}]);
new_ASCII(title_,:)=title_ASCII(1:max_size_title);
m=m+1;
    currtime=toc;
    timeleft(m)=currtime*numeltitles-currtime*title_;
    if rem(title_,1000)==1
        m=0;
        disp(['progress :' num2str(title_/numeltitles*100) '% - time left: ' num2str(nanmean(timeleft)/60) 'm'])
    end
end

validcitations=remaining_validcitations;
validcitationsnorm=remaining_validcitationsnorm;
new_titles_as_ASCII=new_ASCII;

low_25_percent = max(prctile(validcitations,[0 25]));
high_25_percent = min(prctile(validcitations,[75 100]));

idxlow=validcitations<=low_25_percent;
idxhigh=validcitations>=high_25_percent;

low_25_percentsel_cit=validcitations(idxlow);
high_25_percentsel_cit=validcitations(idxhigh);

low_25_percentsel_citnorm=validcitationsnorm(idxlow);
high_25_percentsel_citnorm=validcitationsnorm(idxhigh);

low_25_percentsel_titles=new_titles_as_ASCII(idxlow,:);
high_25_percentsel_titles=new_titles_as_ASCII(idxhigh,:);

mergedvaltitleesel=[[low_25_percentsel_titles;high_25_percentsel_titles],[zeros(size(low_25_percentsel_titles(:,1)));ones(size(high_25_percentsel_titles(:,1)))]];

mergedvalcitesel=[[low_25_percentsel_cit;high_25_percentsel_cit],[zeros(size(low_25_percentsel_cit(:,1)));ones(size(high_25_percentsel_cit(:,1)))]];

mergedvalcitenormsel=[[low_25_percentsel_citnorm;high_25_percentsel_citnorm],[zeros(size(low_25_percentsel_citnorm(:,1)));ones(size(high_25_percentsel_citnorm(:,1)))]];

% save data
disp('saving...')
save('titlesASCII_DictComp','new_titles_as_ASCII')
titlesASCII=new_titles_as_ASCII;
titlesASCII_DictComp=array2table(titlesASCII);
writetable(titlesASCII_DictComp)
save('titlesASCII_low_DictComp','low_25_percentsel_titles')
titlesASCII_low=low_25_percentsel_titles;
titlesASCII_low_DictComp=array2table(titlesASCII_low);
writetable(titlesASCII_low_DictComp)
save('titlesASCII_high_DictComp','high_25_percentsel_titles')
titlesASCII_high=high_25_percentsel_titles;
titlesASCII_high_DictComp=array2table(titlesASCII_high);
writetable(titlesASCII_high_DictComp)
save('titlesASCII_high_low_merged_DictComp','mergedvaltitleesel')
titlesASCII_high_low_merged=mergedvaltitleesel;
titlesASCII_high_low_merged_DictComp=array2table(titlesASCII_high_low_merged);
writetable(titlesASCII_high_low_merged_DictComp)


save('citations_raw_DictComp','validcitations')
citations_raw=validcitations;
citations_raw_DictComp=array2table(citations_raw);
writetable(citations_raw_DictComp)
save('citations_raw_low_DictComp','low_25_percentsel_cit')
citations_raw_low=low_25_percentsel_cit;
citations_raw_low_DictComp=array2table(citations_raw_low);
writetable(citations_raw_low_DictComp)
save('citations_raw_high_DictComp','high_25_percentsel_cit')
citations_raw_high=high_25_percentsel_cit;
citations_raw_high_DictComp=array2table(citations_raw_high);
writetable(citations_raw_high_DictComp)
save('citations_raw_high_low_merged_DictComp','mergedvalcitesel')
citations_raw_high_low_merged=mergedvalcitesel;
citations_raw_high_low_merged_DictComp=array2table(citations_raw_high_low_merged);
writetable(citations_raw_high_low_merged_DictComp)

save('citations_norm_DictComp','validcitationsnorm')
citations_norm=validcitationsnorm;
citations_norm_DictComp=array2table(citations_norm);
writetable(citations_norm_DictComp)
save('citations_norm_low_DictComp','low_25_percentsel_citnorm')
citations_norm_low=low_25_percentsel_citnorm;
citations_norm_low_DictComp=array2table(citations_norm_low);
writetable(citations_norm_low_DictComp)
save('citations_norm_high_DictComp','high_25_percentsel_citnorm')
citations_norm_high=high_25_percentsel_citnorm;
citations_norm_high_DictComp=array2table(citations_norm_high);
writetable(citations_norm_high_DictComp)
save('citations_norm_high_low_merged_DictComp','mergedvalcitenormsel')
citations_norm_high_low_merged=mergedvalcitenormsel;
citations_norm_high_low_merged_DictComp=array2table(citations_norm_high_low_merged);
writetable(citations_norm_high_low_merged_DictComp)

disp('done.')

unix(['sh updateGitHub.sh']);

time_whole_script=toc(time_whole_script)