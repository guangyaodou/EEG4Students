%%
for session = 3;
id = 75; % change subject id
data = load("data_raw/"+id+"_tcr_s"+session+".txt"); 
data = data(1:3000, 2:21);

F=figure;
sgtitle("subject "+id+" session "+session,'fontsize',20)
set(gcf,'color','w');
nodes = ["TP9","AF7","AF8","TP10"];

b=subplot(5,1,2);
plot(data(:,5),'-b')
set(gca,'xtick',[]);
set(subplot(5,1,2),'Color',[0 0 1 0.05])
hold on
% change seperation 
plot(data(:,6)+1.5,'-b')
plot(data(:,7)+2.5,'-b')
plot(data(:,8)+4.1,'-b')
for l = 0:600:2400
    xline(l,'--b')
end
y=ylim;
I=[];
for i=1:4
    I=[I,y(1)+i/5*(y(2)-y(1))];
end
yticks(I)
yticklabels(nodes)
set(gca,'fontsize',16)
text(-450,(y(1)+y(2))/2,"beta\newline13-32HZ",'Color',[0 0 1 0.05],'fontsize',16)
set(gca, 'Position', [.15 0.58 0.8 0.16])

a=subplot(5,1,1);
plot(data(:,13),'-r')
hold on
% change seperation 
plot(data(:,14)+2.7,'-r')
plot(data(:,15)+4.8,'-r')
plot(data(:,16)+7.5,'-r')
set(gca,'xtick',[]);
for l = 0:600:2400
    xline(l,'--r')
end
set(subplot(5,1,1),'Color',[1 0 0 0.05])
y=ylim;
I=[];
for i=1:4
    I=[I,y(1)+i/5*(y(2)-y(1))];
end
yticks(I)
yticklabels(nodes)
set(gca,'fontsize',16)
text(-450,(y(1)+y(2))/2,"gamma\newline32-100HZ",'Color',[1 0 0 0.05],'fontsize',16)

set(gca, 'Position', [0.15 0.74 0.8 0.16])

d=subplot(5,1,4);
plot(data(:,17),'-m')
set(gca,'xtick',[]);
set(subplot(5,1,4),'Color',[1 0 1 0.05])
hold on
% change seperation 
plot(data(:,18)+1.8,'-m')
plot(data(:,19)+2.9,'-m')
plot(data(:,20)+3.5,'-m')
for l = 0:600:2400
    xline(l,'--m')
end
y=ylim;
I=[];
for i=1:4
    I=[I,y(1)+i/5*(y(2)-y(1))];
end
yticks(I)
yticklabels(nodes)
set(gca,'fontsize',16)
text(-450,(y(1)+y(2))/2,"theta\newline4-8HZ",'Color',[1 0 1 0.05],'fontsize',16)
set(gca, 'Position', [.15 0.26 0.8 0.16])

c=subplot(5,1,3);
plot(data(:,1),'-c')
set(gca,'xtick',[]);
set(subplot(5,1,3),'Color',[0 1 1 0.05])
hold on
% change seperation 
plot(data(:,2)+2.8,'-c')
plot(data(:,3)+4.8,'-c')
plot(data(:,4)+6.7,'-c')
for l = 0:600:2400
    xline(l,'--c')
end
y=ylim;
I=[];
for i=1:4
    I=[I,y(1)+i/5*(y(2)-y(1))];
end
yticks(I)
yticklabels(nodes)
set(gca,'fontsize',16)
text(-450,(y(1)+y(2))/2,"alpha\newline8-13HZ",'Color',[0 1 1 0.05],'fontsize',16)
set(gca, 'Position', [.15 0.42 0.8 0.16])

e=subplot(5,1,5);
plot(data(:,9),'-g')
set(gca,'xtick',[]);
set(subplot(5,1,5),'Color',[0 1 0 0.05])
hold on
% change seperation 
plot(data(:,10)+1.6,'-g')
plot(data(:,11)+2.3,'-g')
plot(data(:,12)+3.8,'-g')
for l = 0:600:2400
    xline(l,'--g')
end
y=ylim;
I=[];
for i=1:4
    I=[I,y(1)+i/5*(y(2)-y(1))];
end
yticks(I)
yticklabels(nodes)
set(gca,'fontsize',16)
text(-450,(y(1)+y(2))/2,"delta\newline0.5-4HZ",'Color',[0 1 0 0.05],'fontsize',16)
set(gca, 'Position', [.15 0.1 0.8 0.16])

xticks([600 1200 1800 2400 3000])
xticklabels([1,2,3,4,5])
xlabel("minutes")
% set(F, 'PaperPositionMode', 'auto')
% set(gcf, 'InvertHardCopy', 'off'); % setting 'grid color reset' off
% set(gcf, 'Color', [1 1 1]); %setting figure window background color back to white
%saveas(F,strcat("subject"+subject_id+"_session"+session+"_bandgraph.png"));
% cla;
end
