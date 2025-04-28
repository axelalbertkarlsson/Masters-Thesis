clear;
measurementPath = 'measurement';

addpath(measurementPath)

p = 100;
nSteps = 16; % Number of steps to model cuts/hikes from the central banks


% currency = 'CHF'; 
currency = 'EUR'; 
% currency = 'GBP'; 
% currency = 'JPY'; 
% currency = 'KRW'; 
% currency = 'SEK'; 
% currency = 'USD'; 

mi = marketInfo(currency);

fileName = currency;
% fileName = 'EUR_ICAP';
% fileName = 'EUR_EONIA';
% fileName = 'irsTickEUREONSW=20200203_20200203frq1s';

[pl, times, ric, assetID, assetType, assetTenor, currencies] = populatePortfolioWithData(fileName, mi.currency, mi.currencyTimeZone, mi.iborTimeZone);

indOIS = find(strcmp(mi.onTenor, assetTenor) &  pl.atOISG == assetType);
indIBORON = find(strcmp(mi.onTenor, assetTenor) & pl.atIBOR == assetType);
% indIBORON = [];

isHoliday = mexPortfolio('isHoliday', mi.onCal, floor(times), floor(times)); % Note that projected holidays may change over time (holidays are added and removed, hence the date when the calendar is defined is important)
times(isHoliday==1) = []; % Removes all holidays

[timeData, data] = mexPortfolio('getValues', assetID(indIBORON), times(end), mi.iborTimeZone, 'LAST');
if (floor(timeData) ~= floor(times(end)))
  times(end) = []; % Remove last day, as the IBOR rate do not exist for this
end

if (currency == 'EUR')
  nY = 10; % Number of years the curves span
  times = times(times>= datenum(2005,08,15));

  load EUR100;
  outOfSampleStartDate = datenum(2020,1,1);
  
  cb = readtable("datesEURECB.xlsx", "Sheet", "mHistory");
  
  cbEffectiveDates = datenum(cb.Var11); %CHANGED
  cbDates2 = datenum(cb.Var2); % Dates from ECB  %CHANGED
  ind = ~isnan(cbDates2);
  cbEffectiveDates(ind) = cbDates2(ind); % Replace with actual dates, when rates were actually changed.
elseif (currency == 'SEK')
  nY = 10; % Number of years the curves span
  times = times(times>= datenum(2012,10,22)); % OIS 2-10Y start on this date
elseif (currency == 'USD')
  nY = 10; % Number of years the curves span
  times = times(times>= datenum(2012,03,30)); % OIS 3-10Y start on this date
end

firstDatesPCA = firstDates;
[fH, piH, lastDatesPCA, nIn, ef, Ef, fTotVar, ePi, EPi, piTotVar] = pcaIRS(times, fH, piH, firstDates, lastDates, outOfSampleStartDate);

nH = min(lastDatesPCA-firstDates);
C = cov(fH(:,1:nH));
nEigs = 6;

[V,D] = eigs(C, nEigs);
[ef,ind] = sort(diag(D),1, 'descend');
Ef = V(:,ind);


nD = nY*365+10;

indAll = [indIBORON ; indOIS];
ricAll = [ric(indIBORON) ric(indOIS)];
assetTypeAll = [assetType(indIBORON) ; assetType(indOIS)];

firstDates = ones(length(times), 1)*inf;
tradeDates = ones(length(times), 1)*inf;
lastDates = ones(length(times), 1)*inf;


indInstrAll = cell(length(times), 1);
priceAll = cell(length(times), 1);
atAll = cell(length(times), 1);
tcAll = cell(length(times), 1);
oAll = cell(length(times), 1);
oIndAll = cell(length(times), 1);
usedInstr = false(1, length(indAll));
maturityAll = cell(length(times), 1);

%%& Start: Tillagd av CJ för att ta ut relevant data %%%
TAll = cell(length(times), 1);
T0All = cell(length(times), 1);
%%& End: Tillagd av CJ för att ta ut relevant data %%%


for k=1:length(times)

  
  lastInterestDatesHvec = ones(1, length(indAll))*inf;

  tradeDate = floor(times(k));
%   datestr(tradeDate)

  % IBOR ON
  if (~isempty(indIBORON))
    % Retrieve data
%     [timeData, data] = mexPortfolio('getValues', assetID(indIBORON), times(k), mi.iborTimeZone, 'LAST');

    % RFR rate is set at 08:00:00 the next day, ensure that previous value is used
    rfrTimeLimit = datenum(year(times(k)),month(times(k)),day(times(k)),7, 59, 59);
    [timeData, data] = mexPortfolio('getValues', assetID(indIBORON), rfrTimeLimit, mi.iborTimeZone, 'LAST');
% fprintf('%s: %d\n', datestr(timeData), data);
    rfrTime = datenum(year(timeData), month(timeData), day(timeData));
% fprintf('%s: %s -> %s\n', datestr(times(k)), datestr(rfrTimeLimit), datestr(rfrTime));
    % Create IBOR
    [iborOnID] = mexPortfolio('initFromGenericInterest', assetID(indIBORON), 1, data, rfrTime, 0);
    iborOnPrices = [data ; data ; Inf];
    iborOnMarketQuoteP = data;
  else
    iborOnPrices = zeros(3,0);
    iborOnID = [];
    iborOnMarketQuoteP = [];
  end

  oisID = zeros(size(indOIS));
  oisPrices = ones(3,length(indOIS))*Inf;
  oisMarketQuoteP = ones(1,length(indOIS))*Inf; 
  for j=1:length(indOIS)
    % Retrieve data
    [timeData, data] = mexPortfolio('getValues', assetID(indOIS(j)), times(k), mi.currencyTimeZone, {'BID', 'ASK'});
    if (timeData < times(k)-1) % More than 24 hours old, do not use
      data = [NaN ; NaN];
    end
    maturityDate = mexPortfolio('maturityDate', assetID(indOIS(j)), tradeDate);
    if (maturityDate-tradeDate > nD) % Too long time to maturity, skip asset
      data = [NaN ; NaN];
    end
    oisPrices(1:2,j) = data;
    if (sum(isnan(data))==0)
      mid = mean(data);
      % Create OIS
      [oisID(j)] = mexPortfolio('initFromGenericInterest', assetID(indOIS(j)), 1, mid, tradeDate, 1);
    else
      mid = 0;
      oisID(j) = 1E19; % Removes asset in createInstrumentsClasses
    end
    oisMarketQuoteP(j) = mid;
  end

  instrID = [iborOnID ; oisID];
  prices = [iborOnPrices oisPrices];
  marketQuoteP = [iborOnMarketQuoteP oisMarketQuoteP];
  ricInstr = [ric(indIBORON) ric(indOIS)];
  indInstr = 1:length(instrID);

  conType = ones(size(instrID))*6; % Unique mid price
  [instr,removeAsset,flg] = createInstrumentsClasses(pl, tradeDate, ricInstr, instrID, prices, conType);

  instrID(removeAsset) = [];
  marketQuoteP(removeAsset) = [];
  indInstr(removeAsset) = [];
  indGeneric = indAll(~removeAsset);
  
  firstDate = tradeDate;
  for j=1:length(instrID)
    firstDate = min(firstDate, instr.data{j}.settlementDate);
  end
  indInstrAll{k} = indInstr;

  nExtrapolate = tradeDate-firstDate;
  E = [repmat(Ef(1,:),nExtrapolate, 1) ; Ef];
  datesStep = cbEffectiveDates(cbEffectiveDates > tradeDate);
  datesStep = datesStep(1:nSteps);
  
  Es = zeros(size(E,1), nSteps);
  for i=1:nSteps
    Es(datesStep(i)-tradeDate+1:end, i) = 1;
  end
  E = [E Es];
  intE = [zeros(1, size(E,2)) ; cumsum(E,1)]/365;

  o = [];
  tc = [];
  maturity = zeros(length(instrID),1);
  price = zeros(length(instrID),1);
  oInd = zeros(length(instrID)+1,1);
  oInd(1) = 1;

  %%& Start: Tillagd av CJ för att ta ut relevant data %%%
  T_k = zeros(length(instrID),1);
  T0_k = zeros(length(instrID),1);
  disp(k);
  %%& End: Tillagd av CJ för att ta ut relevant data %%%

  for j=1:length(instrID)
    jj = indInstr(j);
    usedInstr(jj) = true;
    T0 = instr.data{j}.settlementDate - firstDate + 1;
    maturity(j) = instr.data{j}.maturityDate;
    
    if (strcmp(instr.assetType{j},'IBOR'))
      T = instr.data{j}.maturityDate - firstDate + 1;
      o = [o ; -intE(T,:)];
      tc = [tc ; instr.data{j}.dt];
    elseif (strcmp(instr.assetType{j},'OIS'))
      T = instr.data{j}.cfDatesFix - firstDate + 1;
      o = [o ; -intE(T0,:) ; -intE(T,:)];
      tc = [tc ; NaN ; instr.data{j}.dtFix];
    end
    oInd(j+1) = size(o,1)+1;
    price(j) = instr.data{j}.price(3);
    lastInterestDatesHvec(jj) = mexPortfolio('lastDate', instrID(j));
    %%& Start: Tillagd av CJ för att ta ut relevant data %%%
    disp(j);
    if j==17
        disp(T_k);
        disp(size(T));
    end
    T_k(j) = T;
    T0_k(j) = T0;
    %%& End: Tillagd av CJ för att ta ut relevant data %%%
  end
  
  maturityAll{k} = maturity;
  priceAll{k} = price;
  atAll{k} = assetTypeAll(indInstr);
  tcAll{k} = tc;
  oAll{k} = o;
  oIndAll{k} = oInd;

  %%& Start: Tillagd av CJ för att ta ut relevant data %%%
  TAll{k} = T_k;
  T0All{k} = T0_k;
  %%& End: Tillagd av CJ för att ta ut relevant data %%%

  lastDate = mexPortfolio('lastDate', instrID);
 
  firstDates(k) = firstDate;
  tradeDates(k) = tradeDate;
  lastDates(k) = lastDate;
  
end
clearvars -except maturityAll priceAll atAll tcAll oAll oIndAll TAll T0All firstDates tradeDates Ef ef nSteps times fH indInstrAll usedInstr kx pl cbEffectiveDates
