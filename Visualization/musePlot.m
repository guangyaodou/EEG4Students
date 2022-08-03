function [done]= musePlotDataset(data)

s=1;t=0;
    plot(t+data(:,s+0),'-r'); hold on; 
    plot(t+data(:,s+4),'-b'); 
    plot(t+data(:,s+8),'-c'); 
    plot(t+data(:,s+12),'-m'); 
    plot(t+data(:,s+16),'-g');

s=2;t=1;
    plot(t+data(:,s+0),'-r'); hold on; 
    plot(t+data(:,s+4),'-b'); 
    plot(t+data(:,s+8),'-c'); 
    plot(t+data(:,s+12),'-m'); 
    plot(t+data(:,s+16),'-g');

s=3;t=2;
    plot(t+data(:,s+0),'-r'); hold on; 
    plot(t+data(:,s+4),'-b'); 
    plot(t+data(:,s+8),'-c'); 
    plot(t+data(:,s+12),'-m'); 
    plot(t+data(:,s+16),'-g');

s=4;t=3;
    plot(t+data(:,s+0),'-r'); hold on; 
    plot(t+data(:,s+4),'-b'); 
    plot(t+data(:,s+8),'-c'); 
    plot(t+data(:,s+12),'-m'); 
    plot(t+data(:,s+16),'-g');
%title('Plot A: Five bands for four electrodes')
grid on; grid minor;
legend('alpha','beta','delta','gamma','theta');
xticks([0:3000:12000]);
%axis([0,12000,0,100])
hold off;
done = 1.0;
end
