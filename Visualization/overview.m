function [  ] = overview(filename,numClusters)
%overview Show main clustering data for the specified file
%   this calls museplot on all 4 electrodes
%   shows the clusters and their specificity for each of the activities
%   show the 4 meta-clusters and their specificity
%   shows the accuracy of prediction over all the data

windowSize = 600;

% The top plot is the five bands for each of the 4 electrodes

% The second plot is the clusters for each sample

% The third plot is the percentage of a sliding window containing each
% cluster

% The fourth is the number of times that each cluster appears in each of
% the four mental activities, arranged by cluster 

% The fifth is the same number of times that each cluster appears but
% arranged by activity


% Here is the code ..
% read in the data
all = importdata(filename,' ',0);
extra = max(1,round((length(all)-12000)/2))
all = all(extra:length(all)-extra,:);
all = all(:,2:21);

musePlot(all);  % look at all data, select some


% classify 

[idx,X,sumd,D] = kmeans(all,numClusters);

[C,D]=museClassifyAll(all,1,X);

% plot
hold off;

subplot(4,1,1); 
musePlot(all);
xticks([0:3000:12000])
grid on; grid minor;
legend('alpha','beta','delta','gamma','theta');



subplot(4,1,2); 
% plot of the frequency each cluster appears in each core activity
% i.e. the middle three minutes of each activity

a1= hist(C(600:2400),0.5:numClusters-0.5);
a2= hist(C(3600:5400),0.5:numClusters-0.5);
a3= hist(C(6600:8400),0.5:numClusters-0.5);
a4= hist(C(9600:11400),0.5:numClusters-0.5);

aa = [a1;a2;a3;a4]';
bar(aa./30); legend('math','relax1','reading','relax2');
title('Plot B. Frequency each cluster appears in MORC sections')



subplot(4,1,3);
% plot the accuracy of each prediction over all four activities

 vv = (aa' == max(aa'));
 numCC = 4;
 dd = [1:numCC]*vv;
 CC = dd(C);
 c1= hist(CC(1:3000),0.5:numCC-0.5);
 c2= hist(CC(3000:6000),0.5:numCC-0.5);
 c3= hist(CC(6000:9000),0.5:numCC-0.5);
 c4= hist(CC(9000:length(CC)),0.5:numCC-0.5);
 ccs = [c1;c2;c3;c4]'./30;
 bar(ccs'); legend('math','relax1','reading','relax2');
 grid on;
 grid minor;
 axis([0,5,0,100])
 title('Plot C: Accuracy of k-means classification for each MORC region')




subplot(4,1,4); 
zz1 = clusterWindow(CC,windowSize);
plot(zz1./(windowSize/100));
%legend(string(1:max(CC)));
legend('math','relax1','reading','relax2');
grid on; grid minor;
xticks([0:3000:12000])
% look into analyzing data using pca(all) to see which
% vectors have the most importance, and cluster on those??
clusters = [1:numClusters];
mathClusters = clusters(vv(1,:)==1);
openClusters = clusters(vv(2,:)==1);
readClusters = clusters(vv(3,:)==1);
closedClusters = clusters(vv(4,:)==1);
title('Plot D: 1 minute smoothing of k-means classification');


end

