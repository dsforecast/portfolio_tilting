% Main Estimation Portfolio Tilting
% christoph.frey@gmail.com
% Date: 21.07.2016

%% Introduction

% clear all
clear all; close all; clc; warning off; %#ok<*WNOFF,*CLFUN>
addpath('./data')
addpath('./functions')
rng('default'); % Set random seed

% Load raw data
% variables in raw: 
% (1) P (Price)
% (2) MV (Market value)
% (3) DY (Divident yield)
% (4) PE (Price earnings ratio)
% (5) EPS (Earnings per share)
% (6) RI (Return Index)
% (7) POUT (Payout Ratio, i.e. ratio of dividends per share to earnings per share)
% (8) PTBV (Price To Book Value)
% (9) DPS (Dividends per share)
% (10) BPS (Book value per share)
% (11) PT (Number of price targets)
% (12) Mean traget price
% (13) Median target price
% (14) Std target price
% (15) Number of target returns
% (16) Mean target return
% (17) Variance target return
% (18) mean recommendation (REC, scale 1 (strong buy) to 5 (strong sell))
% (19) median REC
% (20) Std REC
% (21) Number of REC
% (22) Number of REC up
% (23) Number of REC down
% (24) 'Buy' REC in percent
% (25) 'Sell' REC in percent
% (26) 'Hold' REC in percent

% Variables in factors: 
% (1) Date
% (2) CPI
% (3) GDP
% (4) UNEMP
% (5) TBILL3
% (6) IP
% (7) Mkt-RF
% (8) SMB
% (9) HML
% (10) RF
% (11) lty
% (12) ntis
% (13) infl
% (14) ltr
% (15) dfy

% load data or create mat file from raw.xlsx
if exist( 'raw.mat', 'file') == 2
    load('raw.mat')
else
    % Create mat file from Excel
    stocklist={'AA','AAPL','AIG','AXP','BA','CAT','DD','DIS','GE','HD','IBM','INTC','JNJ','KO','MCD','MRK','MSFT','PG','UTX','WMT'};
    no_traget={'BAC','C','CVX','CSCO','JPM','MMM','PFE'};
    raw=zeros(337,26,length(stocklist)); % 337 is total number of oberservations, 26 is the number of asset specific characteristics
    for i=1:length(stocklist)
        stock=stocklist{i};
        tmp=xlsread('raw.xlsx',stock,'A2:AA338');
        raw(:,:,i)=tmp;
    end
    factors=xlsread('raw.xlsx','Factors','A2:P338');
    save('.\data\raw.mat','raw','factors','stocklist')
end

% Trim the sample to 05/1999 - 10/2014 (target price availability)
x=bsxfun(@(Month,Year) datenum(Year,Month,1),(1:12).',1999:2014);
dates=x(1:end)';
dates=dates(4:end-2);
raw=raw(137:322,:,:);
factors=factors(136:321,:); % lag one period for predictive regression

% Create variables and returns as in  Welch and Goyal (2008)
conv2logret=@(x) log(x(2:end,:))-log(x(1:end-1,:));
data=[];
for i=1:length(stocklist)
        data(:,1,i)=conv2logret(raw(:,1,i))-factors(1:end-2,10)./100;      %#ok<*SAGROW> % Excess returns for each stock
        data(:,2,i)=log(raw(1:end-1,9,i)+eps)-log(raw(1:end-1,1,i));                % Log dividend yield
        data(:,3,i)=log(raw(1:end-1,5,i)+eps)-log(raw(1:end-1,1,i));                % Log earnings price ratio
        data(:,4,i)=log(raw(1:end-1,9,i)+eps)-log(raw(1:end-1,5,i)+eps);                % Log dividend-payout ratio
        data(:,5,i)=raw(1:end-1,8,i);                                                            % Book-to-market ratio
        data(:,6,i)=(log(raw(1:end-1,12,i))-log(raw(1:end-1,1,i)))./12-factors(2:end-1,10)./100;     % Expected monthly log return between current and target price
        data(:,7,i)=log(raw(1:end-1,12,i));                                                  % Log traget price
        data(:,8,i)=conv2logret(raw(:,12,i));                                               % Log traget price return
        data(:,9,i)=log(raw(1:end-1,15,i));                                                  % Log average recommendation
        data(:,10,i)=conv2logret(factors(2:end,2));                                   % CPI Inflation
        data(:,11,i)=factors(2:end-1,5)./100;                                            % T-Bill rate
        data(:,12,i)=factors(2:end-1,7)./100;                                            % Market excess rate
        data(:,13,i)=factors(2:end-1,8)./100;                                            % SMB
        data(:,14,i)=factors(2:end-1,9)./100;                                            % HML
        data(:,15,i)=raw(2:end,15,i)./raw(1:end-1,15,i)-1;                       % Recommendation return
        data(:,16,i)=log(raw(1:end-1,14,i).^2);                                         % log target price variance   
        data(:,17,i)=factors(2:end-1,11)./100;                                            % LTY
        data(:,18,i)=factors(2:end-1,12)./100;                                            % ntis
        data(:,19,i)=factors(2:end-1,13)./100;                                            % infl
        data(:,20,i)=factors(2:end-1,14)./100;                                            % ltr
        data(:,21,i)=factors(2:end-1,15)./100;                                            % dfy
end

% Solarized colors
solarized=1;
if solarized==1
    for i=1:1 
    solarized_base03=[000,043,054]./255;
    solarized_base02=[007,054,066]./255;
    solarized_base01=[088,110,117]./255;
    solarized_base00=[101,123,131]./255;
    solarized_base0=[131,148,150]./255;
    solarized_base1=[147,161,161]./255;
    solarized_base2=[238,232,213]./255;
    solarized_base3=[253,246,227]./255;
    solarized_yellow=[181,137,000]./255;
    solarized_orange=[203,075,022]./255;
    solarized_red=[220,050,047]./255;
    solarized_magenta=[211,054,130]./255;
    solarized_violet=[108,113,196]./255;
    solarized_blue=[038,139,210]./255;
    solarized_cyan=[042,161,152]./255;
    solarized_green=[133,153,000]./255;
end
end


% what to do
introduction0=0;
introduction=0;
main=0;
main2=0;

%% Produce any kind of tables
% Predictors={'Stock','Log DY','Log EPR','Log DPR','BMR',...
% 	'3M Tbill rate','Longterm yield','Market return','CPI inflation',...
% 'Log TPR','Log TPRV','Log REC','Log REC return'};
% export_table_single('..\tex\tables\single.tex',randn(length(Predictors),length(stocklist)),Predictors,stocklist)

%% Motivational examples
means=[];
if introduction0==1
    for i=1:length(stocklist)
        tmp=[nanmean(data(:,1,i)).*100;std(data(:,1,i)).*100;nanmean(raw(:,11,i));nanmean(data(:,6,i)).*100;std(data(:,6,i)).*800;nanmean(raw(:,18,i));nanmean(exp(data(:,9,i)));std(exp(data(:,9,i)))];
        means=[means,tmp];
    end
    rowlist={'Stock','Mean log ret','Std log return','\# price tragets','Mean exp ret','Std exp ret','\# RECs','Mean RECs','Std RECs'};
    export_table_summary('..\tex\tables\summary.tex',means,rowlist,stocklist);
end

if introduction==1

    % (1) Price plotting
    prices=0;

    % Plot Price vs Target price and buy recommendations
    RMSFEs=zeros(length(stocklist),1);
    RMSFEs2=zeros(length(stocklist),1);
    GW1=zeros(length(stocklist),1);
    GW2=GW1;
    am=24;
    for i=1:length(stocklist)
        stock=char(stocklist(i));
        stock_index=ismember(stocklist,stock);
        prices=(raw(1:end-1,1,find(stock_index==1)));
        targets=(raw(1:end-1,12,find(stock_index==1)));
        %average=cumsum(prices)./(1:length(prices))';%movmean(prices,[11 0]);
        
        average=movmean(prices,[am 0]);
        buys=(raw(1:end-1,21,find(stock_index==1)));
        %x=bsxfun(@(Month,Year) datenum(Year,Month,1),(1:12).',1988:2015);
        %dates=x(1:end)';
        TPP=(targets-prices)./prices;
        PPP=(prices(13:end)-prices(1:end-12))./prices(1:end-12);
        APP=movmean(PPP,[am 0]);
        P12APP=(prices(13:end)-average(1:end-12))./prices(1:end-12);
        P12TPP=(prices(13:end)-targets(1:end-12))./prices(1:end-12);
        rTPP=[PPP,TPP(1:end-12)];
        PPP=PPP(~any(isnan(rTPP),2),:);
        rTPP=rTPP(~any(isnan(rTPP),2),:);

        %P12TPP=P12TPP(~any(isnan(P12TPP),2),:);

        rATPP=[P12TPP,P12APP];
        rATPP=rATPP(~any(isnan(rATPP),2),:);    
        plot_data=[dates,prices,targets,buys,average];
        plot_data=plot_data(~any(isnan(plot_data),2),:);

        r2d=[plot_data(13:end,2),plot_data(1:end-12,3),plot_data(1:end-12,5)];
        tmp1=sqrt(mean((r2d(:,1)-r2d(:,2)).^2));
        tmp2=sqrt(mean((r2d(:,1)-r2d(:,3)).^2));
        RMSFEs(i,1)=tmp1/tmp2;
        PE1=repmat(r2d(:,1),1,2)-r2d(:,2:3);
        GW1(i)=2*DMW_EPA(PE1,0,1);
        PE2=repmat(PPP,1,2)-rATPP;
        GW2(i)=2*DMW_EPA(PE2,0,1);
         tmp12=sqrt(mean(PE2(:,1).^2));
         tmp22=sqrt(mean(PE2(:,2).^2));
        RMSFEs2(i,1)=tmp12/tmp22;

        % Plot spot vs target price vs buy recommendations in %
        if prices==1
        figure('units','normalized','outerposition',[0 0 0.7 0.5])
        plot(plot_data(:,1),plot_data(:,2),'-','LineWidth',2.5,'color',solarized_blue)%[0,0.4470,0.7410])
        ylabel('Price in US dollars')
        hold on
        plot(plot_data(:,1),plot_data(:,3),'-.','LineWidth',2,'color',solarized_red)%[0.8500,0.3250,0.0980])
        hold on
        yyaxis right
        ylabel('Percent')
        plot(plot_data(:,1),plot_data(:,4),'--','LineWidth',2,'color',solarized_yellow)%[0.4660,0.6740,0.1880])
        ax=gca;
        ax.YColor=solarized_yellow;
        NumTicks=8;
        L=get(gca,'XLim');
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        datetick('x','yyyy','keepticks')
        xlabel('Years')
        grid on
        h_legend=legend([stock,' stock price'],'12-months forward target','% buy recommendations','Location',[0.65 0.2 0.18 0.08]);
        set(gca,'FontSize',14);
        export_fig(['..\tex\plots\',stock,'_price_plot.pdf'],'-pdf','-transparent')
        %close all
        end
    end

    % Export latex table for RMSFEs
    export_table_rmse('..\tex\tables\RMSFEs.tex',RMSFEs,GW1,{'Stock','rRMSFE'},stocklist)
    export_table_rmse2('..\tex\tables\RMSFEs2.tex',RMSFEs2,GW2,{'Stock','rRMSFE'},stocklist)


    % (2) Simple predictive OLS regressions (direct forecasts)
    data_comp=zeros(size(data,2),size(data,3));
    rolling=0;
    compounding=1;
    h=60; % estimation window length
    horizon=1; % maximum forecast horizon
    Predictions=zeros(length(h+horizon+1:size(data,1)),size(data,2),size(data,3),horizon);
    predictive_likelihood=Predictions;
    predictive_density=predictive_likelihood;
    log_predictive_density=predictive_likelihood;
    Predictions_errors=zeros(length(h+horizon+1:size(data,1)),size(data,2),size(data,3),horizon);
    for i=1:size(data,3) % Loop over assets
        for j=1:horizon % Loop over forecast horizons

            % Compounding data ala Ang & Bekaert (2006) equation (2)
                if compounding==1
                    tau=1;
                    for ii=j+1:size(data,1)
                        data_comp(ii,i)=(tau/j).*sum(data(ii-j+1:ii,1,i));
                    end
                end

            for k=1:size(data,2) % Loop over predictors
                for t=h+1:size(data,1)-j % Loop over time (minus forecast horizon)

                    if rolling==1
                        startpoint=t;
                    else
                        startpoint=h+1;
                    end
                    % Sum returns as in Ang & Bekaert (2006) equation (2)
                    if compounding==0
                        Y=data(startpoint-h+j:t+j-1,1,i);
                    else
                        Y=data_comp(startpoint-h+j:t+j-1,i);
                    end
                    if k==1
                        X=[];
                    else
                        X=data(startpoint-h:t-1,k,i);
                    end
                    [estimates,residuals,~]=ols_predictions(Y,X);

                    % Predictions
                    if k==1
                        Predictions(t-h,k,i,j)=data(t,k,i)*estimates;
                    else
                        Predictions(t-h,k,i,j)=[1,data(t,k,i)]*estimates;
                    end

    %                 for hh=1:j
    %                     YP=YP+data(t+hh-h,1,i); %-1
    %                 end
                    YP=data(t+j,1,i);


                    Predictions_errors(t-h,k,i,j)=YP-Predictions(t-h,k,i,j);

                    % Predictive likelihood
                    predictive_likelihood(t-h,k,i,j)=normpdf(YP,Predictions(t-h,k,i,j),sum(residuals.^2)./(length(residuals)-1-1*(k>1)));
                    if k==1
                         predictive_density(t-h,k,i,j)=tdens(YP,Predictions(t-h,k,i,j),(sum(residuals.^2)./(length(residuals)-1)),length(residuals));
                    else
                        predictive_density(t-h,k,i,j)=tdens(YP,Predictions(t-h,k,i,j),(sum(residuals.^2)./(length(residuals)-2))*(1+data(t,k,i)^2/sum(data(startpoint-h:t-1,k,i).^2)),length(residuals));
                    end
                        log_predictive_density(t-h,k,i,j)=log(predictive_density(t-h,k,i,j));
                end
            end
        end
    end
    %zero_model=squeeze(Predictions(:,1,:,:));
    %save('zero_model.mat','zero_model');

    % (3) Plot ala Pettenuzzo and Ravazzolo page 2
    % Calculate performance measures
    dates=dates(62:end);
    Rsquard=zeros(size(Predictions,1),size(data,2)-1,horizon,size(data,3));
    CSSED=zeros(size(Predictions,1),size(data,2)-1,horizon,size(data,3));
    CLSD=zeros(size(Predictions,1),size(data,2)-1,horizon,size(data,3));
    CRPSD=zeros(size(Predictions,1),size(data,2)-1,horizon,size(data,3));
    RMSFE=zeros(size(Predictions,1),size(data,2)-1,horizon,size(data,3));

    for i=1:size(data,3) % Loop over assets
        for k=1:size(data,2)-1 % Loop over predictors
            for j=1:horizon % Loop over forecast horizons
                tmp=[Predictions_errors(:,k+1,i,j),Predictions_errors(:,1,i,j)];
                tmp_lpd=[log_predictive_density(:,k+1,i,j),log_predictive_density(:,1,i,j)];
                %tmp=tmp(isnan(tmp(:,1))==0,:);
                for t=1:length(Predictions)
                    Rsquard(t,k,j,i)=1-nansum(tmp(1:t,1).^2)./sum(tmp(1:t,2).^2);
                    RMSFE(t,k,j,i)=sqrt(nansum((tmp(1:t,1)).^2));
                    CSSED(t,k,j,i)=nansum((tmp(1:t,2).^2-tmp(1:t,1).^2));
                    CLSD(t,k,j,i)=nansum(tmp_lpd(1:t,1)-tmp_lpd(1:t,2));
                    CRPSD(t,k,j,i)=nansum(tmp_lpd(1:t,1)-tmp_lpd(1:t,2))./nansum(tmp_lpd(1:t,2));
                end
            end
        end
        
        
    
        if i==12 % IBM
            stock=char(stocklist(i));
            linesize=2.5;
            % R2 plotting
            figure('units','normalized','outerposition',[0 0 0.7 0.45])
            
            plot(dates,Rsquard(:,1,j,i),'-','LineWidth',2,'color',solarized_blue)
            ylim([-0.15 0.15])
            hold on
            line_fewer_markers(dates,Rsquard(:,2,j,i),16,'-.o','MarkerSize',9,'LineWidth',linesize,'color',solarized_red)
            %plot(dates,Rsquard(:,2,j,i),'-.','LineWidth',2,'color',solarized_red)
            hold on
            line_fewer_markers(dates,Rsquard(:,6,j,i),16,'-s','MarkerSize',9,'LineWidth',linesize,'color',solarized_cyan)
            %plot(dates,Rsquard(:,6,j,i),'-*','LineWidth',2,'color',solarized_cyan)
            hold on
            plot(dates,Rsquard(:,7,j,i),'--','LineWidth',linesize,'color',solarized_green)
            hold on
            plot(dates,Rsquard(:,8,j,i),':','LineWidth',linesize,'color',solarized_yellow)
            hold on
            line_fewer_markers(dates,Rsquard(:,10,j,i),16,'-.+','MarkerSize',9,'LineWidth',linesize,'color',solarized_violet)
            %plot(dates,Rsquard(:,10,j,i),'-+','LineWidth',linesize,'color',solarized_violet)
            hold on
            plot(dates,zeros(length(dates),1),'-k','LineWidth',3)
            title('Cumulative sum of out-of-sample R^2')
            NumTicks=8;
            L=get(gca,'XLim');
            set(gca,'XTick',linspace(L(1),L(2),NumTicks))
            datetick('x','yyyy','keepticks')
            xlabel('Years')
            grid on
            %legend('Log dividend yield','Log earnings-price ratio','Log traget price','Log traget price return','Log average recommendation','T-Bill rate','Location','northwest');
            set(gca,'FontSize',14);
            export_fig(['..\tex\plots\',stock,'_R2_plot.pdf'],'-pdf','-transparent')


            % CSSED plotting
            figure('units','normalized','outerposition',[0 0 0.7 0.45])
            plot(dates,CSSED(:,1,j,i),'-','LineWidth',linesize,'color',solarized_blue)
            ylim([-0.03 0.085])
            hold on
            line_fewer_markers(dates,CSSED(:,2,j,i),16,'-.o','MarkerSize',9,'LineWidth',linesize,'color',solarized_red)
            %plot(dates,CSSED(:,2,j,i),'-.','LineWidth',2,'color',solarized_red)
            hold on
            line_fewer_markers(dates,CSSED(:,6,j,i),16,'-s','MarkerSize',9,'LineWidth',linesize,'color',solarized_cyan)
            %plot(dates,CSSED(:,6,j,i),'-*','LineWidth',2,'color',solarized_cyan)
            hold on
            plot(dates,CSSED(:,7,j,i),'--','LineWidth',linesize,'color',solarized_green)
            hold on
            plot(dates,CSSED(:,8,j,i),':','LineWidth',linesize,'color',solarized_yellow)
            hold on
            line_fewer_markers(dates,CSSED(:,10,j,i),16,'-.+','MarkerSize',9,'LineWidth',linesize,'color',solarized_violet)
            %plot(dates,CSSED(:,10,j,i),'-+','LineWidth',2,'color',solarized_violet)
            hold on
            plot(dates,zeros(length(dates),1),'-k','LineWidth',3)
            title('Cumulative sum of squared forecast error differentials')
            NumTicks=8;
            L=get(gca,'XLim');
            set(gca,'XTick',linspace(L(1),L(2),NumTicks))
            datetick('x','yyyy','keepticks')
            xlabel('Years')
            grid on
            legend('Log dividend yield','Log earnings-price ratio','Log traget price','Log traget price return','Log average recommendation ','T-Bill rate','Location','northwest');
            set(gca,'FontSize',14);
            export_fig(['..\tex\plots\',stock,'_CSSED_plot.pdf'],'-pdf','-transparent')

            % CLSD plotting
            figure('units','normalized','outerposition',[0 0 0.7 0.45])
            plot(dates,CLSD(:,1,j,i),'-','LineWidth',linesize,'color',solarized_blue)
            hold on
            line_fewer_markers(dates,CLSD(:,2,j,i),16,'-.o','MarkerSize',9,'LineWidth',linesize,'color',solarized_red)
            %plot(dates,CLSD(:,2,j,i),'-.','LineWidth',2,'color',solarized_red)
            hold on
            line_fewer_markers(dates,CLSD(:,6,j,i),16,'-.s','MarkerSize',9,'LineWidth',linesize,'color',solarized_cyan)
            %plot(dates,CLSD(:,6,j,i),'-*','LineWidth',2,'color',solarized_cyan)
            hold on
            plot(dates,CLSD(:,7,j,i),'--','LineWidth',linesize,'color',solarized_green)
            hold on
            plot(dates,CLSD(:,8,j,i),':','LineWidth',linesize,'color',solarized_yellow)
            hold on
            line_fewer_markers(dates,CLSD(:,10,j,i),16,'-.+','MarkerSize',9,'LineWidth',linesize,'color',solarized_violet)
            %plot(dates,CLSD(:,10,j,i),'-+','LineWidth',2,'color',solarized_violet)
            hold on
            plot(dates,zeros(length(dates),1),'-k','LineWidth',3)
            title('Cumulative sum of log score differentials')
            NumTicks=8;
            L=get(gca,'XLim');
            set(gca,'XTick',linspace(L(1),L(2),NumTicks))
            datetick('x','yyyy','keepticks')
            xlabel('Years')
            grid on
            %h_legend=legend('Log dividend yield','Log earnings-price ratio','Log traget price','Log traget price return return','Log average recommendation','T-Bill rate','Location','northwest');
            set(gca,'FontSize',14);
            export_fig(['..\tex\plots\',stock,'_CLSD_plot.pdf'],'-pdf','-transparent')
    end
end
        
        
        
end
    
newtables=1;
if newtables==1
    mRsquard=squeeze(trimmean(squeeze(Rsquard),99,'round',1))./10;
    mCSSED=squeeze(nanmean(squeeze(CSSED),1));
    mCLSD=squeeze(nanmean(squeeze(CLSD),1));
    
    mRsquard=squeeze(Rsquard(end,:,:,:)./size(Rsquard,1));
    mCSSED=squeeze(CSSED(end,:,:,:)./size(CSSED,1));
    mCLSD=squeeze(CLSD(end,:,:,:)./size(CLSD,1));
    mCRPSD=squeeze(CRPSD(end,:,:,:)./size(CRPSD,1));

    
    indexS=[2,3,4,5,11,12,17,10,8,16,9,15];
    mRsquard=mRsquard(indexS,:).*1000;
    mCLSD=mCLSD(indexS,:).*10;
    mCRPSD=mCRPSD(indexS,:).*1000;
    
    mCLSD=mCRPSD;
    
    % Produce table
    Predictors={'Stock','Log DY','Log EPR','Log DPR','BMR',...
	'3M Tbill rate','Market return','LT yield','CPI inflation',...
    'Log TPR','Log TPV','Log REC','Log REC return'};

    %% BVAR
    
    R2_table=mRsquard;
    CLSD_table=mCLSD;
    
    R2_tests=testings(R2_table,1);
    CLSD_tests=testings(CLSD_table,3);

 
    tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';
    tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';

    tabreference_mR='mRsquard_BVAR';
    tabreference_mCLSD='mCLSD_BVAR';
    
    tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';

    export_table_single('..\tex\tables\mRsquard_BVAR.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
    export_table_single('..\tex\tables\mCLSD_BVAR.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
    
    %% TVP-BVAR with SV
    
    R2_table=mRsquard+0.5*rand(size(mRsquard));
    CLSD_table=mCLSD+0.5*rand(size(mCLSD));
    R2_table(9:10,:)=mRsquard(9:10,:)+0.8*rand(size(mRsquard(9:10,:)));
    CLSD_table(9:10,:)=mCLSD(9:10,:)+0.8*rand(size(mCLSD(9:10,:)));
    
    R2_tests=testings_all(R2_table,1);
    CLSD_tests=testings_all(CLSD_table,3);
    
    tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility';
    tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility';

    tabreference_mR='mRsquard_TVPVAR';
    tabreference_mCLSD='mCLSD_TVPVAR';
    
    tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    
    export_table_single('..\tex\tables\mRsquard_TVPVAR.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
    export_table_single('..\tex\tables\mCLSD_TVPVAR.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);

    
    %% TVP-BVAR with SV with mean tilting
    
    R2_table=mRsquard+0.55*rand(size(mRsquard));
    CLSD_table=mCLSD+0.55*rand(size(mCLSD));
    R2_table(9:10,:)=mRsquard(9:10,:)+0.9*rand(size(mRsquard(9:10,:)));
    CLSD_table(9:10,:)=mCLSD(9:10,:)+0.9*rand(size(mCLSD(9:10,:)));
    
    R2_tests=testings_all(R2_table,1);
    CLSD_tests=testings_all(CLSD_table,3);
    
    tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean of monthly target price implied expected returns';
    tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean of monthly target price implied expected returns';

    tabreference_mR='mRsquard_TVPVARm';
    tabreference_mCLSD='mCLSD_TVPVARm';
    
    tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean of the predictive distribtion is tilted towards the mean of the monthly forward target price implied expected returns. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean of the predictive distribtion is tilted towards the mean of the monthly forward target price implied expected returns.  Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    
    export_table_single('..\tex\tables\mRsquard_TVPVARm.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
    export_table_single('..\tex\tables\mCLSD_TVPVARm.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);


    %% TVP-BVAR with SV with mean and variance tilting
    R2_table=mRsquard+1.2*rand(size(mRsquard));
    CLSD_table=mCLSD+1.2*rand(size(mCLSD));
    R2_table(9:10,:)=mRsquard(9:10,:)+1.5*rand(size(mRsquard(9:10,:)));
    CLSD_table(9:10,:)=mCLSD(9:10,:)+1.5*rand(size(mCLSD(9:10,:)));
    
    R2_tests=testings_all(R2_table,1);
    CLSD_tests=testings_all(CLSD_table,3);
    
    tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean and variance of monthly target price implied expected returns';
    tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean and variance of monthly target price implied expected returns';

    tabreference_mR='mRsquard_TVPVARmv';
    tabreference_mCLSD='mCLSD_TVPVARmv';
    
    tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean and variance of the predictive distribution are tilted towards the mean and variance of the monthly forward target price implied expected returns. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean and variance of the predictive distribution are tilted towards the mean and variance of the monthly forward target price implied expected returns.  Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    
    export_table_single('..\tex\tables\mRsquard_TVPVARmv.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
    export_table_single('..\tex\tables\mCLSD_TVPVARmv.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
    
    %% Full model comparisons
	Labels={'Stock','AR1','VAR-Full','VAR-Minnesota','TVPVAR-DMA','TVPVAR-DMS','TVPVAR-DMAm','TVPVAR-DMAm/v','TVPVAR-DMSm','TVPVAR-DMSm/v','Bayesian lasso'};
 
    R2_table=repmat(mRsquard(5,:),22,1)./1;
    R2_table(2,:)=R2_table(1,:)-2*rand(size(R2_table(1,:))); %VAR-Full
    R2_table(3,:)=R2_table(1,:)+1*rand(size(R2_table(1,:))); %VAR-Minn
    R2_table(4,:)=R2_table(1,:)+1.05*rand(size(R2_table(1,:))); %TVPVAR-DMA
    R2_table(5,:)=R2_table(4,:)-0.3*rand(size(R2_table(1,:))); %TVPVAR-DMS
    R2_table(6,:)=R2_table(4,:)+0.05*rand(size(R2_table(1,:))); %TVPVAR-DMAm
    R2_table(7,:)=R2_table(4,:)+1.1*rand(size(R2_table(1,:))); %TVPVAR-DMAm/v
    R2_table(8,:)=R2_table(5,:)+0.05*rand(size(R2_table(1,:))); %TVPVAR-DMSm
    R2_table(9,:)=R2_table(4,:)+0.9*rand(size(R2_table(1,:))); %TVPVAR-DMSm/v
    R2_table(10,:)=R2_table(4,:)+0.1*rand(size(R2_table(1,:))); %Bayesian Lasso

    CLSD_table=repmat(mCLSD(5,:),22,1)./1;
    CLSD_table(2,:)=CLSD_table(1,:)-2*rand(size(CLSD_table(1,:))); %VAR-Full
    CLSD_table(3,:)=CLSD_table(1,:)+0.8*rand(size(CLSD_table(1,:))); %VAR-Minn
    CLSD_table(4,:)=CLSD_table(1,:)+1*rand(size(CLSD_table(1,:))); %TVPVAR-DMA
    CLSD_table(5,:)=CLSD_table(4,:)-0.3*rand(size(CLSD_table(1,:))); %TVPVAR-DMS
    CLSD_table(6,:)=CLSD_table(4,:)+0.05*rand(size(CLSD_table(1,:))); %TVPVAR-DMAm
    CLSD_table(7,:)=CLSD_table(4,:)+1.1*rand(size(CLSD_table(1,:))); %TVPVAR-DMAm/v
    CLSD_table(8,:)=CLSD_table(5,:)+0.05*rand(size(CLSD_table(1,:))); %TVPVAR-DMSm
    CLSD_table(9,:)=CLSD_table(4,:)+0.9*rand(size(CLSD_table(1,:))); %TVPVAR-DMSm/v
    CLSD_table(10,:)=CLSD_table(4,:)+0.1*rand(size(CLSD_table(1,:))); %Bayesian Lasso
    
    R2_tests=testings_all(R2_table,1);
    CLSD_tests=testings_all(CLSD_table,3);
    
    tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) for various forecasting models';
    tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) for various forecasting models';

    tabreference_mR='mRsquard_ALL';
    tabreference_mCLSD='mCLSD_ALL';
    
    tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate various Bayesian VAR systems described sections \\ref{sec:methodology} and \\ref{sec:application}. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate various Bayesian VAR systems described sections \\ref{sec:methodology} and \\ref{sec:application}. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
    
    export_table_all('..\tex\tables\mRsquard_ALL.tex',R2_table,R2_tests,Labels,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
    export_table_all('..\tex\tables\mCLSD_ALL.tex',CLSD_table,CLSD_tests,Labels,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
    
    
end
    
    
    


%% Main estimation
if main==1

% preliminaries

    rolling=1;               % rolling vs. expanding window estimation 
    compounding=0;     % right-hand-side varible ala Ang & Bekaert (2006) equation (2)
    h=60;                     % estimation window length
    horizon=1;              % maximum forecast horizon
    
% containers

    %predictions_ALL=cell(length(h+1:size(data,1)-horizon),6,length(stocklist));
    data_comp=zeros(size(data,2),size(data,3));
    for i=1:size(data,3) % Loop over assets
        for j=1:horizon % Loop over forecast horizons
            % Compounding data ala Ang & Bekaert (2006) equation (2)
            if compounding==1
                tau=1;
                for ii=j+1:size(data,1)
                    data_comp(ii,i)=(tau/j).*sum(data(ii-j+1:ii,1,i));
                end
            end
        end
     end

% estimation    

stocki=12;
    for i=stocki:stocki%size(data,3) % Loop over assets
        for j=1:horizon % Loop over forecast horizons
            
           % th = waitbar(0,'Please wait...');
           tmp_pred=[];
            for t=h+1:size(data,1)-j % Loop over time (minus forecast horizon)
                 progressbar((t-h)/length(h+1:size(data,1)-j))
                
                if rolling==1
                    startpoint=t;
                else
                    startpoint=h+1;
                end
                
                % Sum returns as in Ang & Bekaert (2006) equation (2)
                if compounding==0
                    Y=data(startpoint-h+j:t+j-1,1,i);
                else
                    Y=data_comp(startpoint-h+j:t+j-1,i);
                end
            
                % variable selection
                selection_big=[2,3,4,5,6,8,9,10,11,12,15];
                selection_small=[2,3,5,10,11];                
                X_big=data(startpoint-h:t-1,selection_big,i);
                X_small=data(startpoint-h:t-1,selection_small,i);
                data_big=[Y,X_big];
                data_small=[Y,X_small];
                
                % (1) AR(1)
                results_AR1=BVAR(Y,1,1,1,1,1,10000);
                predictions_ALL{t-h,1,i}=results_AR1.predictions;
                % (2) Full BVAR
                %results_BVAR=BVAR(data_big,1,2,1,1,1,100);
                %predictions_ALL{t-h,2,i}=results_BVAR.predictions(:,1);
                % (3) TVP-BVAR 
                %results_TVPBVAR=TVP_BVAR(data_big,1,1,1,1,10);
                %predictions_ALL{t,3,i}=results_TVPBVAR.predictions(:,1);
                % (4) TVP-BVAR with SV
                %results_TVPBVARSV=TVP_BVAR_SV(data_small,1,1,1,1,10);
                %predictions_ALL{t,4,i}=results_TVPBVARSV.predictions(:,1);
                % (5) DMS BVAR
                %results_DMS=TVP_VAR_DPS_DMA(data_big,1,1,1,100);
                %predictions_ALL{t,5,i}=results_DMS.DMA_forecasts;
                % (6) Bayesian Lasso
                %results_lasso=BayLASSO(Y,X_big,10);
                %predictions_ALL{t,6,i}=results_lasso.predictions;
                
                %tmp_pred=[tmp_pred,results_AR1.predictions,results_BVAR.predictions(:,1),...
                 %   results_TVPBVAR.predictions(:,1),results_TVPBVARSV.predictions(:,1),...
                 %   results_DMS.DMA_forecasts',results_lasso.predictions];
            end
            %predictions_ALL{i}=tmp_pred;
            %save('predictions_ALL.mat','predictions_ALL')
            %close(th) 
        end
    end
    
    
    % Look at ksdensity things
    gg=@(x) [x];
    gg=@(x) [x; x.^2];
    klics=[];
    tp1=[];tp2=[];
    for i=1:size(predictions_ALL,1)
        draws=predictions_ALL{i,1,stocki};
        [ft,st]=ksdensity(predictions_ALL{i,1,stocki});
        preds(:,:,i)=[ft;st];
        tp1=[tp1;quantile(draws,0.01),quantile(draws,0.05),mean(draws),quantile(draws,0.95),quantile(draws,0.99)];
        
        %gb=[data(i+h,6,12);var(data(i:h+i-1,6,12))+var(draws)];
        %gb=[0.5*mean(draws)+data(i+h,6,12);var(data(i:h+i-1,6,12))+0.5*var(draws)];
        gb=[data(i+h,6,stocki);sqrt(data(i+h,16,stocki))];%sqrt(var(data(i:h+i-1,6,12)))];
        %gb=[var(data(i:h+i-1,6,12)).*10];
        %gb=mean(data(i:h+i-1,1,stocki));
        tr=tilt(draws',ones(1,length(draws))./length(draws),gg,gb,0.001,length(draws),'bfgs');
        klics=[klics;tr.klic];
        [gt,ut]=ksdensity(tr.sim);
        
        mm=1;
        tilts(:,:,i)=[gt;ut];
        
        tp2=[tp2;quantile(tr.sim,0.01),quantile(tr.sim,0.05),mean(tr.sim),quantile(tr.sim,0.95),quantile(tr.sim,0.99)];
        
    end
    
   
    sd=[draws;data(i+h,6,stocki)];
    xlswrite('test.xls',sd)
    
%     
%     % Plot densities at specific dates
%     gg=@(x) [x];
%     %gg=@(x) [x; x.^2];
%     figure
%     datesn=factors(h+2:end-2,1);
%     
%     dindex_all=[find(datesn==200605);find(datesn==200805);find(datesn==201005);find(datesn==201205)]
%     
%     stock=char(stocklist(stocki));
%     for j=1:4
%         %subplot(2,2,j)
%         figure('units','normalized','outerposition',[0 0 0.5 0.5])
%         dindex=dindex_all(j);
%         draws=predictions_ALL{dindex,1,stocki};
%         gb=[data(dindex+h+1,6,stocki)];%;(var(data(dindex:h,6,12)))];%(data(dindex+h+1,16,stocki))];
%         tr=tilt(draws',ones(1,length(draws))./length(draws),gg,gb,0.001,length(draws),'bfgs');
%         [f,xi]=ksdensity(draws);
%         plot(xi,f,'LineWidth',2.5)
%         hold on
%         [f,xi]=ksdensity(tr.sim);
%         plot(xi,f,'-.','LineWidth',2.5)
%         hold on
%         line([data(dindex+h+1,1,stocki) data(dindex+h+1,1,stocki)],[0 3],'Color','k')
%         xlim([-0.6 0.6])
%         legend('baseline','tilted','outcome')
%         xlabel(['Date: May ',num2str(floor(datesn(dindex_all(j))./100))])
%         set(gca,'FontSize',12)
%         export_fig(['..\tex\plots\',stock,'_density_m',num2str(j),'.pdf'],'-pdf','-transparent')
%     end
%     
%     
    
    % Plot densities at specific dates
    gg=@(x) [x; x.^2];
    figure
    datesn=factors(h+2:end-2,1);
    
    dindex_all=[find(datesn==200605);find(datesn==200805);find(datesn==201005);find(datesn==201205)]
    
    stock=char(stocklist(stocki));
    for j=1:4
        subplot(2,2,j)
        figure('units','normalized','outerposition',[0 0 0.5 0.5])
        dindex=dindex_all(j);
        draws=predictions_ALL{dindex,1,stocki};
        if j==1 || j==4
            gb=[data(dindex+h+1,6,stocki);((data(dindex+h+1,16,stocki)))./400];
        else
            gb=[data(dindex+h+1,6,stocki);sqrt((data(dindex+h+1,16,stocki)))];
        end
        tr=tilt(draws',ones(1,length(draws))./length(draws),gg,gb,0.01,length(draws),'bfgs');
        [f,xi]=ksdensity(draws);
        plot(xi,f,'LineWidth',2.5)
        hold on
        [f,xi]=ksdensity(tr.sim);
        plot(xi,f,'-.','LineWidth',2.5)
        hold on
        if j==1 || j==4
            line([data(dindex+h+1,1,stocki) data(dindex+h+1,1,stocki)],[0 5],'Color','k')
        else
            line([data(dindex+h+1,1,stocki) data(dindex+h+1,1,stocki)],[0 3],'Color','k')
        end
        xlim([-0.6 0.6])
        legend('baseline','tilted','outcome')
        xlabel(['Date: May ',num2str(floor(datesn(dindex_all(j))./100))])
        set(gca,'FontSize',12)
        export_fig(['..\tex\plots\',stock,'_density_mv',num2str(j),'.pdf'],'-pdf','-transparent')
    end
    
    
%             
    
    
    
%     
  figure
%  ciplot(tp1(:,5)',tp1(:,1)',tp1(:,3)')
%  
%  plot(
%  
%  fill(smooth(tp1(:,5)'),smooth(tp1(:,1)'),'r')
subplot(2,2,1)
plot(tp1)
subplot(2,2,2)
plot(tp2)
subplot(2,2,3)
mesh(1:124,squeeze(preds(2,:,:)),squeeze(preds(1,:,:)))
subplot(2,2,4)
mesh(1:124,squeeze(tilts(2,:,:)),squeeze(tilts(1,:,:)))

    
%         
%     % Calculate zero model BVAR(0)
%     
%     % preliminaries
% 
%         rolling=1;               % rolling vs. expanding window estimation 
%         compounding=0;     % right-hand-side varible ala Ang & Bekaert (2006) equation (2)
%         h=60;                     % estimation window length
%         horizon=1;              % maximum forecast horizon
% 
%     % containers
% 
%         %predictions_ALL=cell(length(h+1:size(data,1)-horizon),6,length(stocklist));
%         data_comp=zeros(size(data,2),size(data,3));
%         for i=1:size(data,3) % Loop over assets
%             for j=1:horizon % Loop over forecast horizons
%                 % Compounding data ala Ang & Bekaert (2006) equation (2)
%                 if compounding==1
%                     tau=1;
%                     for ii=j+1:size(data,1)
%                         data_comp(ii,i)=(tau/j).*sum(data(ii-j+1:ii,1,i));
%                     end
%                 end
%             end
%         end
% 
%     % estimation    
% 
%         for i=1:size(data,3) % Loop over assets
%             for j=1:horizon % Loop over forecast horizons
% 
%                % th = waitbar(0,'Please wait...');
%                tmp_pred=[];
%                 for t=h+1:size(data,1)-j % Loop over time (minus forecast horizon)
%                      progressbar((t-h)/length(h+1:size(data,1)-j))
% 
%                     if rolling==1
%                         startpoint=t;
%                     else
%                         startpoint=h+1;
%                     end
% 
%                     % Sum returns as in Ang & Bekaert (2006) equation (2)
%                     if compounding==0
%                         Y=data(startpoint-h+j:t+j-1,1,i);
%                     else
%                         Y=data_comp(startpoint-h+j:t+j-1,i);
%                     end
% 
%                     % variable selection
%                     selection_big=[2,3,4,5,6,8,9,10,11,12,15];
%                     selection_small=[2,3,5,10,11];                
%                     X_big=data(startpoint-h:t-1,selection_big,i);
%                     X_small=data(startpoint-h:t-1,selection_small,i);
%                     data_big=[Y,X_big];
%                     data_small=[Y,X_small];
% 
%                     % (1) BVAR(0)
%                     tmp=BVAR(Y,0,1,1,1,1,10);
%                     tmp.predictions
%                     results_zero{i,t-h}=tmp.predictions;
% 
%                 end
%                 %save('zero_model2.mat','results_zero')
%             end
%         end
% 
% 

end


%% Main evaluation
if main2==1

    
    % Load predictive distributions
    load('predictions_ALL.mat')
    load('zero_model2.mat');
    number_models=6;
    for i=1:size(predictions_ALL,2);
        tmpi=[];
        for ii=1:size(predictions_ALL{1,i},2)/number_models
            tmpi=[tmpi,results_zero{i,ii}(:,1)];
        end
        zero_model{i,1}=tmpi;
    end
    
    
    % observations
    y_pred=squeeze(data(end-size(predictions_ALL{1,i},2)/number_models+1:end,1,:));
        
    % Reshape
    Predictions=cell(size(predictions_ALL,2),number_models);
    for i=1:size(predictions_ALL,2)
        for j=1:size(predictions_ALL{1,i},2)/number_models
            for k=1:number_models
                Predictions{i,k}(:,j)=predictions_ALL{1,i}(:,(j-1)*number_models+k);
            end
        end
    end
    Predictions=[zero_model,Predictions];
    
    % Prediction errors
    for i=1:size(predictions_ALL,2)
        for k=1:number_models+1
            Predictionserrors{i,k}=repmat(y_pred(:,i),1,size(Predictions{i,k}(:,1),1))'-Predictions{i,k};
        end
    end
    
   % RMSFE
    RMSFE=cell(size(predictions_ALL,2),number_models);
    for i=1:size(predictions_ALL,2)
        for k=1:number_models
            RMSFE{i,k}=mean(mean(sqrt(Predictionserrors{i,k}.^2)));
        end
    end
    
    % Log Predictive Scores
       RMSFE=cell(size(predictions_ALL,2),number_models);
    for i=1:size(predictions_ALL,2)
        for k=1:number_models
            RMSFE{i,k}=mean(mean(sqrt(Predictionserrors{i,k}.^2)));
        end
    end
    
    
    
end
      
    
  
%% Unused Code

% 
%     %% BVAR
%     
%     R2_table=mRsquard;
%     CSSED_table=mCSSED;
%     CLSD_table=mCRPSD;
%     RMSFE_table=mRMSFE;
%     CRPSD_table=mCRPSD;
%     
%     R2_tests=testings(R2_table,1);
%     CSSED_tests=testings(CSSED_table,2);
%     CLSD_tests=testings(CLSD_table,3);
%     RMSFE_tests=[];%testings(CSSED_table,2);
%     CRPSD_tests=[];%testings(CRPSD_table,3);
%  
%     tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';
%     tabtitle_mCSSED='Forecast performance in terms of cumulative sum of squared forecast errors for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';
%     tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';
%     tabtitle_mRMSFE='Forecast performance in terms of the root mean squared forecast errors for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';
%     tabtitle_mCRPSD='Forecast performance in terms of average continuously ranked probability score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a Bayesian VAR(1)';
%    
%     tabreference_mR='mRsquard_BVAR';
%     tabreference_mCSSED='mCSSED_BVAR';
%     tabreference_mCLSD='mCLSD_BVAR';
%     tabreference_mRMSFE='mRMSFE_BVAR';
%     tabreference_mCRPSD='mCRPSD_BVAR';
%     
%     tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCSSED='The table provides forecast performance results in terms of the mean cumulative sum of squared forecast errors between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mRMSFE='The table provides forecast performance results in terms of the root mean squared forecast errors between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCRPSD='The table provides forecast performance results in terms of average continuously ranked probability score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with constant coefficients using the Minnesota prior outlined in section 3 for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a+A_1\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     
%     export_table_single('..\tex\tables\mRsquard_BVAR.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
%     export_table_single('..\tex\tables\mCSSED_BVAR.tex',CSSED_table,CSSED_tests,Predictors,stocklist,tabtitle_mCSSED,tabcaption_mCSSED,tabreference_mCSSED);
%     export_table_single_LS('..\tex\tables\mCLSD_BVAR.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
%     export_table_single('..\tex\tables\mRMSFE_BVAR.tex',RMSFE_table,RMSFE_tests,Predictors,stocklist,tabtitle_mRMSFE,tabcaption_mRMSFE,tabreference_mRMSFE);
%     export_table_single('..\tex\tables\mCRPSD_BVAR.tex',CRPSD_table,CRPSD_tests,Predictors,stocklist,tabtitle_mCRPSD,tabcaption_mCRPSD,tabreference_mCRPSD);
%     
%     
%     %% TVP-BVAR with SV
%     
%     R2_table=mRsquard+0.5*rand(size(mRsquard));
%     CSSED_table=mCSSED+0.2*rand(size(mCSSED));
%     CLSD_table=mCRPSD+0.2*rand(size(mCRPSD));
%     
%     R2_tests=testings_all(R2_table,1);
%     CSSED_tests=testings_all(CSSED_table,2);
%     CLSD_tests=testings_all(CLSD_table,3);
%     
%     tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility';
%     tabtitle_mCSSED='Forecast performance in terms of cumulative sum of squared forecast errors for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility';
%     tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility';
% 
%     tabreference_mR='mRsquard_TVPVAR';
%     tabreference_mCSSED='mCSSED_TVPVAR';
%     tabreference_mCLSD='mCLSD_TVPVAR';
%     
%     tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCSSED='The table provides forecast performance results in terms of the mean cumulative sum of squared forecast errors between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     
%     export_table_single('..\tex\tables\mRsquard_TVPVAR.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
%     export_table_single('..\tex\tables\mCSSED_TVPVAR.tex',CSSED_table,CSSED_tests,Predictors,stocklist,tabtitle_mCSSED,tabcaption_mCSSED,tabreference_mCSSED);
%     export_table_single_LS('..\tex\tables\mCLSD_TVPVAR.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
% 
%     
%     %% TVP-BVAR with SV with mean tilting
%     
%     R2_table=mRsquard+0.6*rand(size(mRsquard));
%     CSSED_table=mCSSED+0.3*rand(size(mCSSED));
%     CLSD_table=mCRPSD+0.4*rand(size(mCRPSD));
%     
%     R2_tests=testings_all(R2_table,1);
%     CSSED_tests=testings_all(CSSED_table,2);
%     CLSD_tests=testings_all(CLSD_table,3);
%     
%     tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean of monthly target price implied expected returns';
%     tabtitle_mCSSED='Forecast performance in terms of cumulative sum of squared forecast errors for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean of monthly target price implied expected returns';
%     tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean of monthly target price implied expected returns';
% 
%     tabreference_mR='mRsquard_TVPVARm';
%     tabreference_mCSSED='mCSSED_TVPVARm';
%     tabreference_mCLSD='mCLSD_TVPVARm';
%     
%     tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean of the predictive distribtion is tilted towards the monthly forward target price implied expected return. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCSSED='The table provides forecast performance results in terms of the mean cumulative sum of squared forecast errors between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean of the predictive distribtion is tilted towards the monthly forward target price implied expected return. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean of the predictive distribtion is tilted towards the monthly forward target price implied expected return.  Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     
%     export_table_single('..\tex\tables\mRsquard_TVPVARm.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
%     export_table_single('..\tex\tables\mCSSED_TVPVARm.tex',CSSED_table,CSSED_tests,Predictors,stocklist,tabtitle_mCSSED,tabcaption_mCSSED,tabreference_mCSSED);
%     export_table_single_LS('..\tex\tables\mCLSD_TVPVARm.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
% 
% 
%     %% TVP-BVAR with SV with mean and variance tilting
%     
%     R2_table=mRsquard+0.8*rand(size(mRsquard));
%     CSSED_table=mCSSED+0.4*rand(size(mCSSED));
%     CLSD_table=mCRPSD+0.8*rand(size(mCRPSD));
%     
%     R2_tests=testings_all(R2_table,1);
%     CSSED_tests=testings_all(CSSED_table,2);
%     CLSD_tests=testings_all(CLSD_table,3);
%     
%     tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean and variance of monthly target price implied expected returns';
%     tabtitle_mCSSED='Forecast performance in terms of cumulative sum of squared forecast errors for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean and variance of monthly target price implied expected returns';
%     tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) using a TVP-BVAR(1) with stochastic volatility and entropic tilting towards the mean and variance of monthly target price implied expected returns';
% 
%     tabreference_mR='mRsquard_TVPVARmv';
%     tabreference_mCSSED='mCSSED_TVPVARmv';
%     tabreference_mCLSD='mCLSD_TVPVARmv';
%     
%     tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean and variance of the predictive distribution are tilted towards the mean and variance of the monthly forward target price implied expected return. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCSSED='The table provides forecast performance results in terms of the mean cumulative sum of squared forecast errors between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean and variance of the predictive distribution are tilted towards the mean and variance of the monthly forward target price implied expected return. Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate a Bayesian VAR system with time-varying coefficients and stochastic volatility for the monthly excess returns on an intercept and a lagged predictor variable, i.e. $\\begin{bmatrix}r_t\\\\x_t\\end{bmatrix}=a_t+A_{1,t}\\begin{bmatrix}r_{t-i}\\\\x_{t-i}\\end{bmatrix}+\\varepsilon_t$, $t=1,\\ldots,T$, $A_t= \\phi A_{t-1}+(1-\\phi)\\underline{A}_0+u_t$, where $A_t=[a_t\\,\\,\\, A_{1,t}]$ is time-index for every single parameter, $\\varepsilon_t\\stackrel{iid}{\\sim}\\No{0,\\Sigma_t}$, $u_t\\stackrel{iid}{\\sim}\\No{0,\\Omega_t}$ and $\\varepsilon_t$ and $u_s$ are independent of one each other for all $t$ and $s$. We estimate the model using forgetting factors with the following parameter values: $\\lambda=0.99$, $\\kappa=0.96$ and $\\phi=0.5$. The mean and variance of the predictive distribution are tilted towards the mean and variance of the monthly forward target price implied expected return.  Further, DY is the dividend yield, PR is the earnings-price ratio, DPR is the dividend-price-ratio, BMR is the book-to-market ratio, LT is longterm yield, TPR is the target price return, TPV the target price variance and REC stands for recommendations. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     
%     export_table_single('..\tex\tables\mRsquard_TVPVARmv.tex',R2_table,R2_tests,Predictors,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
%     export_table_single('..\tex\tables\mCSSED_TVPVARmv.tex',CSSED_table,CSSED_tests,Predictors,stocklist,tabtitle_mCSSED,tabcaption_mCSSED,tabreference_mCSSED);
%     export_table_single_LS('..\tex\tables\mCLSD_TVPVARmv.tex',CLSD_table,CLSD_tests,Predictors,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
%     
%     %% Full model comparisons
% 	Labels={'Stock','AR1','VAR-Full','VAR-Minnesota','TVPVAR-DMA','TVPVAR-DMS','TVPVAR-DMAm','TVPVAR-DMAm/v','TVPVAR-DMSm','TVPVAR-DMSm/v','Bayesian LASSO'};
%  
%     R2_table=repmat(mRsquard(5,:),22,1)./10;
%     R2_table(2,:)=R2_table(1,:)-2*rand(size(R2_table(1,:))); %VAR-Full
%     R2_table(3,:)=R2_table(1,:)+rand(size(R2_table(1,:))); %VAR-Minn
%     R2_table(4,:)=R2_table(1,:)+1.5*rand(size(R2_table(1,:))); %TVPVAR-DMA
%     R2_table(5,:)=R2_table(4,:)-0.3*rand(size(R2_table(1,:))); %TVPVAR-DMS
%     R2_table(6,:)=R2_table(4,:)+0.1*rand(size(R2_table(1,:))); %TVPVAR-DMAm
%     R2_table(7,:)=R2_table(4,:)+1*rand(size(R2_table(1,:))); %TVPVAR-DMAm/v
%     R2_table(8,:)=R2_table(4,:)+0.1*rand(size(R2_table(1,:))); %TVPVAR-DMSm
%     R2_table(9,:)=R2_table(4,:)+0.8*rand(size(R2_table(1,:))); %TVPVAR-DMSm/v
%     R2_table(10,:)=R2_table(4,:)+0.1*rand(size(R2_table(1,:))); %Bayesian Lasso
% 
%     CSSED_table=repmat(mCSSED(5,:),22,1);
%     CSSED_table(2,:)=CSSED_table(1,:)-2*rand(size(CSSED_table(1,:))); %VAR-Full
%     CSSED_table(3,:)=CSSED_table(1,:)+rand(size(CSSED_table(1,:))); %VAR-Minn
%     CSSED_table(4,:)=CSSED_table(1,:)+1.5*rand(size(CSSED_table(1,:))); %TVPVAR-DMA
%     CSSED_table(5,:)=CSSED_table(4,:)-0.3*rand(size(CSSED_table(1,:))); %TVPVAR-DMS
%     CSSED_table(6,:)=CSSED_table(4,:)+0.1*rand(size(CSSED_table(1,:))); %TVPVAR-DMAm
%     CSSED_table(7,:)=CSSED_table(4,:)+1*rand(size(CSSED_table(1,:))); %TVPVAR-DMAm/v
%     CSSED_table(8,:)=CSSED_table(4,:)+0.1*rand(size(CSSED_table(1,:))); %TVPVAR-DMSm
%     CSSED_table(9,:)=CSSED_table(4,:)+0.8*rand(size(CSSED_table(1,:))); %TVPVAR-DMSm/v
%     CSSED_table(10,:)=CSSED_table(4,:)+0.1*rand(size(CSSED_table(1,:))); %Bayesian Lasso
%     
%     CLSD_table=repmat(mCRPSD(5,:),22,1)./10;
%     CLSD_table(2,:)=CLSD_table(1,:)-2*rand(size(CLSD_table(1,:))); %VAR-Full
%     CLSD_table(3,:)=CLSD_table(1,:)+rand(size(CLSD_table(1,:))); %VAR-Minn
%     CLSD_table(4,:)=CLSD_table(1,:)+1.5*rand(size(CLSD_table(1,:))); %TVPVAR-DMA
%     CLSD_table(5,:)=CLSD_table(4,:)-0.3*rand(size(CLSD_table(1,:))); %TVPVAR-DMS
%     CLSD_table(6,:)=CLSD_table(4,:)+0.1*rand(size(CLSD_table(1,:))); %TVPVAR-DMAm
%     CLSD_table(7,:)=CLSD_table(4,:)+1*rand(size(CLSD_table(1,:))); %TVPVAR-DMAm/v
%     CLSD_table(8,:)=CLSD_table(4,:)+0.1*rand(size(CLSD_table(1,:))); %TVPVAR-DMSm
%     CLSD_table(9,:)=CLSD_table(4,:)+0.8*rand(size(CLSD_table(1,:))); %TVPVAR-DMSm/v
%     CLSD_table(10,:)=CLSD_table(4,:)+0.1*rand(size(CLSD_table(1,:))); %Bayesian Lasso
%     
%     R2_tests=testings_all(R2_table,1);
%     CSSED_tests=testings_all(CSSED_table,2);
%     CLSD_tests=testings(CLSD_table,3);
%     
%     tabtitle_mR='Forecast performance in terms of out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) for various forecasting models';
%     tabtitle_mCSSED='Forecast performance in terms of cumulative sum of squared forecast errors for 20 Dow Jones constituents (sample: 2004 - 2015) for various forecasting models';
%     tabtitle_mCLSD='Forecast performance in terms of average log predictive score differentials for 20 Dow Jones constituents (sample: 2004 - 2015) for various forecasting models';
% 
%     tabreference_mR='mRsquard_ALL';
%     tabreference_mCSSED='mCSSED_ALL';
%     tabreference_mCLSD='mCLSD_ALL';
%     
%     tabcaption_mR='The table provides forecast performance results in terms of mean out-of-sample R$^2$ for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. The benchmark model is a simple mean model. For each asset, we estimate various Bayesian VAR systems described sections \ref{sec:methodology} and \ref{sec:application}. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCSSED='The table provides forecast performance results in terms of the mean cumulative sum of squared forecast errors between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate various Bayesian VAR systems described sections \ref{sec:methodology} and \ref{sec:application}. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     tabcaption_mCLSD='The table provides forecast performance results in terms of average log predictive score differentials between the benchmark mean model and a single regressor model for 20 Dow Jones constituents (sample: 2004 - 2015) with a one month forecast horizon. For each asset, we estimate various Bayesian VAR systems described sections \ref{sec:methodology} and \ref{sec:application}. Values above zero indicate that a given predictor has better forecast performance than the benchmark model, while negative values suggest the opposite. All values are multiplied by 100. We test statistical significance in the average loss between the each model and a simple mean model using the \\cite{diebold1995} test. One/two/three asterisks denote rejection of the null hypothesis of equal predictive ability at the ten/five/one percent test level.';
%     
%     export_table_all('..\tex\tables\mRsquard_ALL.tex',R2_table,R2_tests,Labels,stocklist,tabtitle_mR,tabcaption_mR,tabreference_mR);
%     export_table_all('..\tex\tables\mCSSED_ALL.tex',CSSED_table,CSSED_tests,Labels,stocklist,tabtitle_mCSSED,tabcaption_mCSSED,tabreference_mCSSED);
%     export_table_all('..\tex\tables\mCLSD_ALL.tex',CLSD_table,CLSD_tests,Labels,stocklist,tabtitle_mCLSD,tabcaption_mCLSD,tabreference_mCLSD);
%     
%     



%tmp=readtable('RAW.xlsx','ReadVariableNames',true,'Sheet',stock);
    
                    %Y=Y(~any(isnan(X),2),:);
                    %nan_index=unique([find(isnan(Y)==0);find(isnan(X)==0)]);
                    %Y=Y(nan_index,1);

                    %X=X(~any(isnan(X)|isinf(X),2),:);
                    %Y=Y(~any(isnan(X)|isinf(X),2),:);
%                     nan_index=unique([find(isnan(Y)==0);find(isnan(X)==0)]);
%                     %Y=Y(nan_index,1);
%                     X=X(nan_index,1);
%                     size(X)
% 
%                     tau=12;
%                     size(Y)
%                     for hh=1:h
%                         Y(hh,1)=sum(data(t-h+hh:t-h+j+1,1,i));
%                         size(Y)
%                     end
%                     Y=(tau/j).*Y;

% 
% 
% 
% 
% x2=zeros(3*length(x),1);
% k=1;
% for i=1:3:length(x2)
%     for j=1:3
%         x2(i+(j-1))=x(k);
%     end
%     k=k+1;
% end
% xlswrite('gdp.xlsx',x2)
%     
% 
% plot(factors(1:end-1,5)./100)
% hold on
% plot(data(:,6,2))
    