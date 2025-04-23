function [instruments,removeAsset,flg] = createInstrumentsClasses(pl, tradeDate, rics, assetID, prices, conType)

%ricInd = [indCD indFRA indIRS]; % Ska egentligen vara inargument n�r
                                % instrument ric kan klassificeras
%conType = ones(size(ricInd))*6; % Ska egentligen vara inargument

% conType = 1 - Spread, 2 - Spread bid, 3 - Spread ask, 
%           4 - Unique bid, 5 - Unique ask, 6 - Unique mid

removeAsset = zeros(size(assetID));
instruments = clsInstruments;

for i = 1:length(assetID)

  bid = prices(1,i);
  ask = prices(2,i);
  hasBid = ~isnan(bid);
  hasAsk = ~isnan(ask);

  if (~(hasBid || hasAsk) || ...
      (conType(i) == 2 && ~hasBid) || ...
      (conType(i) == 3 && ~hasAsk) || ...
      (conType(i) == 4 && ~hasBid) || ...
      (conType(i) == 5 && ~hasAsk) || ...
      (conType(i) == 6 && ~(hasBid && hasAsk)) || ...
      assetID(i) > 1E18) % FxSwap ON or TN did not exist
    removeAsset(i) = 1;
    continue; % Skip current RIC because of lack of data
  end

  if (conType(i) == 1)
    price = [bid ask inf];
  elseif (conType(i) == 2)
    price = [bid inf inf];
  elseif (conType(i) == 3)
    price = [-inf ask inf];
  elseif (conType(i) == 4)
    price = [-inf inf bid];
  elseif (conType(i) == 5)
    price = [-inf inf ask];
  elseif (conType(i) == 6)
    price = [-inf inf (bid+ask)/2];
  end

  % Make sure that prices do not contain nan
  if (isnan(price(1)))
    price(1) = -inf;
  end
  if (isnan(price(2)))
    price(2) = inf;
  end
  if (isnan(price(3)))
    price(3) = inf;
  end

  assetType = mexPortfolio('assetType', assetID(i)); 
  if (assetType == pl.atFxSwap)
    fxSwap = clsFxSwap(rics{i}, assetID(i));
    fxSwap.tradeDate = tradeDate;
    fxSwap.price=price;
    instruments.add(fxSwap, 'FxSwap', rics{i}, fxSwap.settlementDate, fxSwap.maturityDate);
  elseif (assetType == pl.atDeposit)
    ibor = clsIBOR(rics{i}, assetID(i));
    ibor.tradeDate = tradeDate;
    ibor.price=price;
    instruments.add(ibor, 'IBOR', rics{i}, ibor.settlementDate, ibor.maturityDate);
  elseif (assetType == pl.atFuture)
    future = clsFutureIR(rics{i}, assetID(i));
    price(3) = (100-price(3))/100; % Quickfix
    future.price = price;
    instruments.add(future, 'FRA', rics{i}, future.valueDate, future.maturityDate);
  elseif (assetType == pl.atFutureCommodity)
    futureCommodity = clsFutureCommodity(rics{i}, assetID(i), tradeDate);
    if (futureCommodity.maturityDate <= tradeDate)
      removeAsset(i) = 1;
      continue;
    end
    price(3) = price(3); 
    futureCommodity.price = price;
    instruments.add(futureCommodity, 'FRA', rics{i}, futureCommodity.tradeDate, futureCommodity.maturityDate);
  elseif (assetType == pl.atFRA)
    fra = clsFRA(rics{i}, assetID(i));
    fra.price = price;
    instruments.add(fra, 'FRA', rics{i}, fra.valueDate, fra.maturityDate);
  elseif (assetType == pl.atIRS)
    irs = clsIRS(rics{i}, assetID(i));
    irs.price = price;
    instruments.add(irs, 'IRS', rics{i}, irs.settlementDate, irs.maturityDate);
  elseif (assetType == pl.atOIS)
    ois = clsOIS(rics{i}, assetID(i));
    ois.price = price;
    instruments.add(ois, 'OIS', rics{i}, ois.settlementDate, ois.maturityDate);
  else
    error('Not able to deal with instrument of type %d', assetType)
  end

end

removeAsset = (removeAsset == 1); % Need to convert to logical indexing
flg = 0;
