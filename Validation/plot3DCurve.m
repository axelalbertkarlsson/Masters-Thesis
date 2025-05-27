function plot3DCurve(times, maturities, fAll, title)
% --- Synthetic data creation ---
% Create a range of dates from 2015 to 2020, spaced 30 days apart
% times = datenum(2015,1,1):30:datenum(2020,1,1);
TT = length(times);  % Number of date points

% Create maturities, say 1 to 10 years
% maturities = 1:10;
N = length(maturities);

% Initialize cell array for forward rates
% fAll = cell(TT,1);

% Populate forward rates with some made-up pattern
% For example, a gentle upward slope in maturities
% plus a seasonal sinusoidal over time, just to illustrate
% for t = 1:TT
%     % E.g. base + slope * maturities + time sinusoid
%     baseLevel = 1.0;  
%     slope     = 0.2;  
%     timeFactor = 0.5 * sin(2 * pi * t / TT);  
%     fAll{t} = baseLevel + slope .* maturities + timeFactor; 
% end

% --- Construct a matrix from the cell array ---
% Rows = different dates
% Columns = different maturities
fMatrix = fAll;

% --- Create grids for plotting ---
% Let X = maturities, Y = times (or vice versa, your choice)
[X, Y] = meshgrid(maturities, times);

% --- 3D surface plot ---
surf(X, Y, fMatrix');   % If dimension mismatch, use surf(X, Y, fMatrix') or transpose X/Y/fMatrix
shading interp;        % smooth shading
colormap(jet);
colorbar;
xlabel('Maturity (Years)');
ylabel('Date (Serial Days)');
zlabel('Forward Rate (%)');
sgtitle(sprintf('Synthetic Forward Rate Surface - %s', title));

% If your "times" are serial date numbers, format them nicely on the Y-axis
datetick('y','yyyy');  % convert Y tick marks to readable years

axis tight;
view([-45 30]);        % adjust 3D view angle
end
