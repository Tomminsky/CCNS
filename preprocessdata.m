%%% data preprocessing %%%

clc
clear 
fileID=fopen('/Users/Tommy/Documents/Nijmegen/Study/CCNS/final_project/allcrawspreproc.txt');
cd('/Users/Tommy/Documents/Nijmegen/Study/CCNS/final_project/')
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

%% save data
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

%% for dictionary represented titles (NEEDS STILL TO BE DONE)
clc
clear
load('titlesASCII')
%%
dictionary_titles_raw=char(new_titles_as_ASCII);
dictionary_titles=arrayfun(@(x) strsplit(dictionary_titles_raw(x,:)),1:size(dictionary_titles_raw,1),'unif',0);

dictionary=[dictionary_titles{:}];

[dictionaryunique, ~, J]=unique(dictionary);
occurance = histc(J, 1:numel(dictionaryunique));



%%

tic
wordidx=arrayfun(@(x) find(strcmp(dictionary_titles{1}{x},dictionaryunique)),1:size(dictionary_titles{1},2));
toc

titles_as_dictionaryindices=nan(size(new_titles_as_ASCII));

numeltitles=numel(dictionary_titles);


progress=1/numeltitles;
timeleft=nan(100,1);
actualidx=1:numel(dictionaryunique);
m=0;

dictionaryuniquethresh=dictionaryunique(occurance>1);

for n=1:numeltitles
    tic
    wordidx=arrayfun(@(x) find(strcmp(dictionary_titles{n}{x},dictionaryuniquethresh)),1:size(dictionary_titles{n},2),'unif',0);
    wordidx=[wordidx{:}];
    sizewordidx=size(wordidx,2);
    titles_as_dictionaryindices(n,1:sizewordidx)=wordidx;
    titles_as_dictionaryindices(n,sizewordidx+1:end)=zeros;
    currtime=toc;
    
    m=m+1;
    timeleft(m)=currtime*numeltitles-currtime*n;
    if rem(n,100)==1
        m=0; 
        disp(['progress :' num2str(n/numeltitles*100) '% - time left: ' num2str(nanmean(timeleft)/60) 'm'])
    end
end

save('dictionary','dictionaryuniquethresh')
dictionary=dictionaryuniquethresh';
dictionary=cell2table(dictionary);
writetable(dictionary)

save('titlesDict','titles_as_dictionaryindices')
titlesDict=titles_as_dictionaryindices;
titlesDict=array2table(titlesDict);
writetable(titlesDict)