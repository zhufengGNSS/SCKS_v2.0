function pcolornum(C,L,CM,FS,conf,A)

% C  is correlation matrix
% L  labels (cell of strings in column {'A';'B';'C'} 
% CM color map e.g. hot
% FS  font size

if nargin<4 || isempty(FS)
 FS = 10; 
end
if nargin<3 || isempty(CM)
    if sum(any(C<0))
        h = hot(64);
        h(1:8,:) = [];
        h2 = [h(:,3),h(:,2),h(:,1)];
        CM = [h2;flipud(h)];
    else
        h = hot;
        h(1:8,:) = [];
        CM = flipud(h);
    end
end
if nargin<2 || isempty(L)
 L = num2str([1:size(C,1)]'); 
end

cf = reshape([1:size(C,1)^2]',size(C,1),size(C,1));
cf = rot90(cf');
cf = cf(:);
C = rot90(C');
C0 = zeros(size(C,1)+1);
C0(1:size(C,1),1:size(C,1)) = C;
if sum(any(C<0))
    C0(end,end) = -1*max(abs(C0(:)));
    C0(1,end) = max(abs(C0(:)));
end
gcf;
pcolor(C0);            % Create a colored plot of the matrix values
colormap(CM);  % Change the colormap to gray (so higher values are
                         %   black and lower values are white)

textStrings = num2str(C(:),'%0.2f');  % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x,y] = meshgrid(1:size(C,1));   % Create x and y coordinates for the strings
hStrings = text(x(:)+0.5,y(:)+0.5,textStrings(:),...      %Plot the strings
                'HorizontalAlignment','center','FontSize',FS);
            x = x(:);
            y = y(:);
            Cc = C;
            midValue = get(gca,'CLim')/2;  % Get the middle value of the color range
            Cc(Cc>midValue(2))=1;
            Cc(Cc<midValue(1))=1;
Cc(Cc~=1)=0;
textColors = repmat(Cc(:),1,3);  % Choose white or black for the
                                             %   text color of the strings so
                                             %   they can be easily seen over
                                             %   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  % Change the text colors

if nargin==5            
for i=1:length(x)
rectangle('Position',[x(i),y(i),conf(cf(i)),conf(cf(i))],'Curvature',[1,1],...
          'FaceColor','none','EdgeColor',textColors(i,:))
end
end

if nargin==6            
for i=1:length(x)
if A(cf(i))
rectangle('Position',[x(i),y(i),A(cf(i)),A(cf(i))],'Curvature',[0,0],...
          'FaceColor','none','EdgeColor',[0 0.9 0],'linewidth',3)
end
end
end

axis square
colorbar

set(gca,'XTick',(1:size(C,1))+0.5,...                         % Change the axes tick marks
        'XTickLabel',L,...  %   and tick labels
        'YTick',(1:size(C,1))+0.5,...
        'YTickLabel',flipud(L),...
        'TickLength',[0 0]);
