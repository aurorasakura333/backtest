import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class FactorModel:
    """Class to calculate and manage market factors like MKT, SMB, HML"""
    
    def __init__(self, data: pd.DataFrame):

        self.data = data
        self.factors = {}
        self.factor_returns = pd.DataFrame()
        
    def calculate_factors(self):
        """Calculate all factors and factor returns"""
        self.calculate_market_factor()
        self.calculate_size_factor()
        self.calculate_value_factor()
        return self.factor_returns
    
    def calculate_market_factor(self):
        """
        Calculate market factor (MKT) - market return minus risk-free rate
    
        """
        # Group by date and calculate average returns
        market_returns = self.data.groupby('date')['returns'].mean()
        
        # Risk-free rate (in a real implementation, use actual risk-free rate data)
        # Here we use a simple constant approximation
        risk_free_rate = 0.0001  # Daily risk-free rate (approximately 2.5% annually)
        
        # Calculate market factor (excess return)
        mkt_factor = market_returns - risk_free_rate
        
        # Store in factors dictionary
        self.factors['MKT'] = mkt_factor
        
        # Add to factor returns DataFrame
        self.factor_returns['MKT'] = mkt_factor
        
        return mkt_factor
    
    def calculate_size_factor(self):
        """
        Calculate size factor (SMB - Small Minus Big)

        """
        if 'market_cap' not in self.data.columns:
            raise ValueError("Market cap data required for size factor calculation")
        
        dates = self.data['date'].unique()
        smb_returns = []
        
        for date in dates:
            # Get data for current date
            date_data = self.data[self.data['date'] == date].copy()
            
            if len(date_data) < 10:  # Skip dates with insufficient data
                smb_returns.append(np.nan)
                continue
            
            # Calculate median market cap
            median_market_cap = date_data['market_cap'].median()
            
            # Create small and big portfolios
            small_stocks = date_data[date_data['market_cap'] <= median_market_cap]
            big_stocks = date_data[date_data['market_cap'] > median_market_cap]
            
            # Calculate portfolio returns (equal-weighted)
            small_return = small_stocks['returns'].mean()
            big_return = big_stocks['returns'].mean()
            
            # Calculate SMB factor
            smb = small_return - big_return
            smb_returns.append(smb)
        
        # Create Series with dates as index
        smb_factor = pd.Series(smb_returns, index=dates)
        
        # Store in factors dictionary
        self.factors['SMB'] = smb_factor
        
        # Add to factor returns DataFrame
        self.factor_returns['SMB'] = smb_factor
        
        return smb_factor
    
    def calculate_value_factor(self):
        """
        Calculate value factor (HML - High Minus Low)
 
        """
        if 'btm' not in self.data.columns:
            raise ValueError("Book-to-market data required for value factor calculation")
        
        dates = self.data['date'].unique()
        hml_returns = []
        
        for date in dates:
            # Get data for current date
            date_data = self.data[self.data['date'] == date].copy()
            
            if len(date_data) < 10:  # Skip dates with insufficient data
                hml_returns.append(np.nan)
                continue
            
            # Calculate 30th and 70th percentiles of book-to-market ratio
            low_percentile = date_data['btm'].quantile(0.3)
            high_percentile = date_data['btm'].quantile(0.7)
            
            # Create value and growth portfolios
            high_btm_stocks = date_data[date_data['btm'] >= high_percentile]
            low_btm_stocks = date_data[date_data['btm'] <= low_percentile]
            
            # Calculate portfolio returns (equal-weighted)
            high_return = high_btm_stocks['returns'].mean()
            low_return = low_btm_stocks['returns'].mean()
            
            # Calculate HML factor
            hml = high_return - low_return
            hml_returns.append(hml)
        
        # Create Series with dates as index
        hml_factor = pd.Series(hml_returns, index=dates)
        
        # Store in factors dictionary
        self.factors['HML'] = hml_factor
        
        # Add to factor returns DataFrame
        self.factor_returns['HML'] = hml_factor
        
        return hml_factor
    
    def calculate_momentum_factor(self, lookback_period: int = 12, skip_recent: int = 1):
        """
        Calculate momentum factor (UMD - Up Minus Down)
        
        Args:
            lookback_period: Number of months to look back for momentum calculation
            skip_recent: Number of months to skip (to avoid short-term reversals)
            
        Momentum is typically calculated as:
        1. Sort stocks by past returns (looking back lookback_period months, skipping skip_recent months)
        2. Form winner (W) and loser (L) portfolios based on past returns
        3. UMD = return of winners - return of losers
        """
        dates = self.data['date'].unique()
        umd_returns = []
        
        # First, calculate past returns for each stock on each date
        for date in dates:
            # This simplified implementation assumes daily data
            # In a real implementation, you'd use monthly data and proper lookback windows
            date_data = self.data[self.data['date'] == date].copy()
            
            if len(date_data) < 10:  # Skip dates with insufficient data
                umd_returns.append(np.nan)
                continue
            
            # Since we may not have past return data ready, we'll just use a simplified approach
            # In practice, you'd calculate proper lookback returns for each stock
            
            # Use 30/70 percentiles to define winners and losers
            winner_threshold = date_data['returns'].quantile(0.7)
            loser_threshold = date_data['returns'].quantile(0.3)
            
            winners = date_data[date_data['returns'] >= winner_threshold]
            losers = date_data[date_data['returns'] <= loser_threshold]
            
            winner_return = winners['returns'].mean()
            loser_return = losers['returns'].mean()
            
            umd = winner_return - loser_return
            umd_returns.append(umd)
        
        # Create Series with dates as index
        umd_factor = pd.Series(umd_returns, index=dates)
        
        # Store in factors dictionary
        self.factors['UMD'] = umd_factor
        
        # Add to factor returns DataFrame
        self.factor_returns['UMD'] = umd_factor
        
        return umd_factor
    
    def calculate_factor_exposures(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate factor exposures (betas) for a given return series
        
        Args:
            returns: Series of returns (must have same index as factor returns)
            
        Returns:
            Dictionary of factor exposures
        """
        # Make sure factors have been calculated
        if self.factor_returns.empty:
            self.calculate_factors()
        
        # Align indices
        aligned_returns = returns.reindex(self.factor_returns.index)
        
        # Create DataFrame with returns and factors
        regression_data = pd.concat([aligned_returns, self.factor_returns], axis=1)
        regression_data.columns = ['returns'] + list(self.factor_returns.columns)
        
        # Drop rows with missing values
        regression_data = regression_data.dropna()
        
        # Calculate factor exposures using OLS regression
        X = regression_data[self.factor_returns.columns]
        X = sm.add_constant(X)  # Add constant for intercept
        y = regression_data['returns']
        
        model = sm.OLS(y, X).fit()
        
        # Extract exposures
        exposures = {factor: model.params[factor] for factor in self.factor_returns.columns}
        exposures['alpha'] = model.params['const']
        exposures['r_squared'] = model.rsquared
        
        return exposures
    
    def calculate_factor_contributions(self, returns: pd.Series) -> pd.DataFrame:
        """
        Calculate factor contributions to returns
        
        Args:
            returns: Series of returns (must have same index as factor returns)
            
        Returns:
            DataFrame with factor contributions
        """
        # Calculate factor exposures
        exposures = self.calculate_factor_exposures(returns)
        
        # Calculate factor contributions
        contributions = pd.DataFrame(index=self.factor_returns.index, columns=list(self.factor_returns.columns) + ['alpha', 'total'])
        
        for factor in self.factor_returns.columns:
            contributions[factor] = exposures[factor] * self.factor_returns[factor]
        
        contributions['alpha'] = exposures['alpha']
        contributions['total'] = contributions[list(self.factor_returns.columns) + ['alpha']].sum(axis=1)
        
        return contributions

# Helper function to prepare data for factor models
def prepare_data_for_factors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for factor model calculations
    
    Args:
        data: Raw DataFrame
        
    Returns:
        Processed DataFrame ready for factor calculations
    """
    # Make a copy
    df = data.copy()
    
    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and code
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # Calculate returns
    df['returns'] = df.groupby('code')['close'].pct_change()
    
    # If market_cap is not available, estimate it using price * shares_outstanding
    if 'market_cap' not in df.columns and 'shares_outstanding' in df.columns:
        df['market_cap'] = df['close'] * df['shares_outstanding']
    
    # If btm (book-to-market) is not available, it needs to be added from external data
    
    return df