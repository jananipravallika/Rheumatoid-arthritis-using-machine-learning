clear all; 
close all; 
clc;
warning off all
%% General settings
fontSize = 14;
%%%%%%%%   stage 1 Image acquisition   %%%%%%%%%%%%%%

%% Pick the image and Load the image
 [filename, pathname] = uigetfile( ...
       {'*.jpg;*.tif;*.tiff;*.png;*.bmp', 'All image Files (*.jpg, *.tif, *.tiff, *.png, *.bmp)'}, ...
        'Pick a file');
f = fullfile(pathname, filename);
disp('Reading image')
rgbImage = imread(f);
rgbImage = imresize(rgbImage,[256 256]);
[rows columns numberOfColorBands] = size(rgbImage);


%%%%%%%%%%%%%%%  checking this image is as gray or rgb range

% % % % %        image conversion process

if strcmpi(class(rgbImage), 'uint8')
		% Flag for 256 gray levels.
		eightBit = true;
	else
		eightBit = false;
    end
    
	if numberOfColorBands == 1
		if isempty(storedColorMap)
			rgbImage = cat(3, rgbImage, rgbImage, rgbImage);
		else
			% It's an indexed image.
			rgbImage = ind2rgb(rgbImage, storedColorMap);
			if eightBit
				rgbImage = uint8(255 * rgbImage);
			end
		end
    end      	
    
%% Display the color image 
disp('Displaying color original image')
F1 = figure(1);
   subplot(3,4,1);
    imshow(rgbImage);
    
    if numberOfColorBands > 1 
		title('Image Acquisition', 'FontSize', fontSize); 
	else 
		caption = sprintf('Original Indexed Image\n(converted to true color with its stored colormap)');
		title(caption, 'FontSize', fontSize);
    end
  
%% Size of the picture - to occupy the whole screen
 scnsize = get(0,'ScreenSize'); % - - width height
 position = get(F1,'Position'); % x-pos y-pos widht height
 outerpos = get(F1,'OuterPosition');
 borders = outerpos - position;
 edge = abs(borders(1))/2;
 pos1 = [edge,...
        1/20*scnsize(4), ...
        9/10*scnsize(3),...
        9/10*scnsize(4)];
 set(F1,'OuterPosition',pos1) 

%% Explore RGB    
% Extract out the color bands from the original image
% into 3 separate 2D arrays, one for each color component.
	redBand = rgbImage(:, :, 1); 
    redBandSmooth=medfilt2(redBand);
    redBand=imadjust(redBandSmooth);
	greenBand = rgbImage(:, :, 2); 
    greenBandSmooth=medfilt2(greenBand);
    greenBand=imadjust(greenBandSmooth);
	blueBand = rgbImage(:, :, 3); 
    blueBandSmooth=medfilt2(blueBand);
    blueBand=imadjust(blueBandSmooth);
   % Display them.
	subplot(3, 4, 2);
        imshow(redBand);
        title('Enhancement Stage-1', 'FontSize', fontSize);
	subplot(3, 4, 3);
        imshow(greenBand);
        title('Enhancement Stage-2', 'FontSize', fontSize);
	subplot(3, 4, 4);
        imshow(blueBand);
        title('Image Filteration', 'FontSize', fontSize);

%% Compute and plot the red histogram. 
	hR = subplot(3, 4, 6); 
	[countsR, grayLevelsR] = imhist(redBand); 
	maxGLValueR = find(countsR > 0, 1, 'last'); 
	maxCountR = max(countsR); 
	bar(countsR, 'r'); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	title('Enhancement-1 Histogram', 'FontSize', fontSize);

%% Compute and plot the green histogram. 
	hG = subplot(3, 4, 7); 
	[countsG, grayLevelsG] = imhist(greenBand); 
	maxGLValueG = find(countsG > 0, 1, 'last'); 
	maxCountG = max(countsG); 
	bar(countsG, 'g', 'BarWidth', 0.95); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	title('Enhancement-2 Histogram', 'FontSize', fontSize);

%% Compute and plot the blue histogram. 
	hB = subplot(3, 4, 8); 
	[countsB, grayLevelsB] = imhist(blueBand); 
	maxGLValueB = find(countsB > 0, 1, 'last'); 
	maxCountB = max(countsB); 
	bar(countsB, 'b'); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	title('Filteration Histogram', 'FontSize', fontSize);

%% Set all axes to be the same width and height.
% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]); 
	if eightBit 
			maxGL = 255; 
	end 
	maxCount = max([maxCountR,  maxCountG, maxCountB]); 
	axis([hR hG hB], [0 maxGL 0 maxCount]); 

%% Plot all 3 histograms in one plot.
	subplot(3, 4, 5); 
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	hold on; 
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2); 
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2); 
	title('Image Histogram', 'FontSize', fontSize); 
	maxGrayLevel = max([maxGLValueR, maxGLValueG, maxGLValueB]); 
	% Trim x-axis to just the max gray level on the bright end. 
	if eightBit 
		xlim([0 255]); 
	else 
		xlim([0 maxGrayLevel]); 
    end
    
redMask=im2bw(redBand,0.4);
greenMask=~(im2bw(greenBand,0.2));
blueMask=im2bw(blueBand,0.7);
    
%% Display the thresholded binary images.
	subplot(3, 4, 10);
        imshow(redMask, []);
        title('Smoothening', 'FontSize', fontSize);
	subplot(3, 4, 11);
        imshow(greenMask, []);
        title('Intensity Adjustment', 'FontSize', fontSize);
	subplot(3, 4, 12);
        imshow(blueMask, []);
        title('Gamma Correction', 'FontSize', fontSize);
	    
%% Combine the masks to find where all 3 are "true."
% Then we will have the mask of only the chosen color parts of the image.
% 
ObjectsMask=uint8(redMask | greenMask | blueMask);
	subplot(3, 4, 9);
	imshow(ObjectsMask, []);
	caption = sprintf('Edge Filteration');
	title(caption, 'FontSize', fontSize);

% ObjectsMask=greenMask;
    ObjectsMaskao = uint8(bwareaopen(ObjectsMask,40));
  figure
    subplot(2,2,1)
    imshow(ObjectsMaskao, []);
    title('Erosion', 'FontSize', fontSize)   
    subplot(2,2,2);
        ObjectsMaskfill = uint8(imfill(ObjectsMask, 'holes'));
        imshow(ObjectsMaskfill, []);
	title('Dialation', 'FontSize', fontSize);  
    Se=strel('disk',1);
    ObjectsMaskerode=imerode(ObjectsMaskfill,Se);
   subplot(2,2,3);
   imshow(ObjectsMaskerode, []);
	title('Final Image Correction', 'FontSize', fontSize);  
    
Area_all=regionprops(ObjectsMaskerode,'Area');
[v,x,y,z]=bwboundaries(ObjectsMaskerode);
 subplot(2,2,4);
 imshow(rgbImage);
 title('Highlighting Boundaries','FontSize', fontSize);
 hold on; 
 for k = 1:length(v)
     boundary = v{k};
     plot(boundary(:,2),boundary(:,1),'r','linewidth',1.5);
 end  
 
OurIm=zeros(size(rgbImage));
 for i=1:size(redMask,1)
     for j=1:size(redMask,2)
         if ObjectsMaskerode(i,j)==1
             OurIm(i,j,:)=rgbImage(i,j,:);
         end
     end
 end
 
 figure;imshow(uint8(OurIm));title('Segmented Image');

 img=double(imresize(rgbImage,[100 100]));
 [s1,s2,s3]=size(img);
X1 = img(:,:,1); 
X2 = img(:,:,2); 
X3 = img(:,:,3);  
X = [X1(:) X2(:) X3(:)]; % [(s1*s2)x3]
k = 4; % no. of clusters

CostFunction=@(m) ClusteringCost2(m, X);     % Cost Function m = [3x2] cluster centers

VarSize=[k size(X,2)];  % Decision Variables Matrix Size = [4 3]

nVar=prod(VarSize);     % Number of Decision Variables = 12

VarMin= repmat(min(X),1,k);      % Lower Bound of Variables [4x1] of[1x3] = [4x3]
VarMax= repmat(max(X),1,k);      % Upper Bound of Variables [4x1] of[1x3] = [4x3]

% Setting the Gaussian Rule
ga_opts = gaoptimset('display','iter');

% running the Gaussian with desired options
[centers, err_ga] = ga(CostFunction, nVar,[],[],[],[],VarMin,VarMax,[],[],[]);

m=centers;

    % Calculate Distance Matrix
    g=reshape(m,3,4)'; % create a cluster center matrix(4(clusters) points in 3(features) dim plane)=[4x3]
    d = pdist2(X, g); % create a distance matrix of each data points in input to each centers = [(s1*s2)x4]

    % Assign Clusters and Find Closest Distances
    [dmin, ind] = min(d, [], 2);
    % ind value gives the cluster number assigned for the input = [(s1*s2)x1]
    
    % Sum of Within-Cluster Distance
    WCD = sum(dmin); 
    
    z=WCD; % fitness function contain sum of each data point to their corresponding center value set (aim to get it minimum)    
    % z = [1 x 1]     

outimg=reshape(ind,s1,s2);
outimgN=img;
    for i=1:s1
        for j=1:s2
            if outimg(i,j)== 1
                outimgN(i,j,:)= [240,36,148];
            elseif outimg(i,j)== 2
                outimgN(i,j,:)= [0,0,255];
            elseif outimg(i,j)== 3
                outimgN(i,j,:)= [255,230,10];
            elseif outimg(i,j)== 4
                outimgN(i,j,:)= [255,0,0];
            end
        end
    end
    figure;imshow(uint8(outimgN));
    title('Representation with Thermal View');
    
%     ObjectsMaskerode
   ObjectsMaskerode=ModelFile(greenMask); 
    FinoutSegment=imresize(outimg,[256 256]).*double(ObjectsMaskerode);
%     figure;imshow(FinoutSegment,[]);
%     title('Final Segmented image');
    OutSegrgb=zeros(size(rgbImage));
    for i=1:size(redMask,1)
     for j=1:size(redMask,2)
         if FinoutSegment(i,j)>0
             OutSegrgb(i,j,:)=rgbImage(i,j,:);
         end
     end
    end
    
    [newim,mas]=rgb2ind(OutSegrgb,3);
    %% Statistical and Textural Feature Extraction
    newim=newim*255;
g = graycomatrix(newim);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(newim);
Standard_Deviation = std2(newim);
Entropy = entropy(newim);
RMS = mean2(rms(newim));
Variance = mean2(var(double(newim)));
a = sum(double(newim(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(newim(:)));
Skewness = skewness(double(newim(:)));
% Inverse Difference Movement
m = size(newim,1);
n = size(newim,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = newim(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
%% Geometrical Feature Extraction    
[val, num]=bwlabel(newim);
STATSGeometry=regionprops(val,'All');
AreaA=0;
PerimeterA=0;
CircularityA=0;
CentroidA=[];
for iol=1:num
    AreaA=AreaA+STATSGeometry(iol,1).Area;
    PerimeterA=PerimeterA+STATSGeometry(iol,1).Perimeter;
    CentroidA=[CentroidA ;STATSGeometry(iol,1).Centroid];
    CircularityA =CircularityA+ (4*pi*AreaA)/(PerimeterA^2);
end
CentroidMidPointX=sum(CentroidA(:,1))/num;
CentroidMidPointY=sum(CentroidA(:,2))/num;
 
featTest = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM, AreaA, PerimeterA, CircularityA,CentroidMidPointX,CentroidMidPointY];
Loinf=find(featTest==Inf);
if ~isempty(Loinf)
featTest(Loinf)=1;
end
load Databasen1_fin
extracted_feature=featTrain(:,1:end);
train_label23={1,2,3,4,5,4,5,3,1,4,5}; 
train_label={[1;1],2,[3;3],[4;4;4],[5;5;5]}; 
train_cell={extracted_feature([1 9],:),extracted_feature(2,:),extracted_feature([3 8],:),...
    extracted_feature([4 6 10],:),extracted_feature([5 7 11],:)}; 
[svmstruct,level] = Train_SVM(train_cell,train_label); 
label=[1 2 3 4 5]; 
[Class_test] = Classify_SVM(featTest,label,svmstruct,level);

%  correlation for second decision
for iop=1:11
    Cout(iop)=corr2(newim,FeatIm(iop).Matrix);
end
[Mco,Mind]=max(Cout);
Class_test=train_label23{Mind};
if Class_test==1 
    Resultdisease='Mid Level of Affection';
elseif Class_test==2 
    Resultdisease='Early Stage of Affection';
elseif Class_test==3 
    Resultdisease='Early Stage of Affection';
else
    Resultdisease='High Level of Affection';
end    

OututFinal=['Prediction Status : ' Resultdisease];
msgbox(OututFinal);