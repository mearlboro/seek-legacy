{-# LANGUAGE DeriveDataTypeable, GADTs, MultiParamTypeClasses, OverloadedStrings, StandaloneDeriving, TypeFamilies #-}
import Control.Monad
import Data.Hashable
import Data.Typeable
import Haxl.Core
import Text.Printf


data Knowledge a where
    Knowledge :: String -> Knowledge String
  deriving Typeable

data Topic a where
    Topic1 :: String -> Topic String
    Topic2 :: String -> Topic String
  deriving Typeable

runKnowledge :: Knowledge a -> ResultVar a -> IO ()
runKnowledge (Knowledge x ) var = putSuccess var (printf "Knowledge ( %s ) " x )

runTopic :: Topic a -> ResultVar a -> IO ()
runTopic (Topic1 x) var = putSuccess var (printf " Topic1 ( %s ) " x)
runTopic (Topic2 x) var = putSuccess var (printf " Topic2 ( %s ) " x)

deriving instance Show (Knowledge a)
deriving instance Show (Topic a)
deriving instance Eq (Knowledge a)
deriving instance Eq (Topic a)

instance DataSourceName Knowledge where
    dataSourceName _ = "Knowledge"
instance DataSourceName Topic where
    dataSourceName _ = "Topic"

instance Show1 Knowledge where
    show1 (Knowledge x) = printf "Knowledge ( %s ) " x 
instance Show1 Topic where
    show1 (Topic1 x) = printf "Topic1 ( %s ) " x 
    show1 (Topic2 x) = printf "Topic2 ( %s ) " x

instance Hashable (Knowledge a) where
    hashWithSalt salt (Knowledge x) = hashWithSalt salt (x)
instance Hashable (Topic a) where
    hashWithSalt salt (Topic1 x) = hashWithSalt salt (1 :: Int, x)
    hashWithSalt salt (Topic2 x) = hashWithSalt salt (2 :: Int, x)

instance StateKey Knowledge where
    data State Knowledge = NoStateKnowledge
instance StateKey Topic where
    data State Topic = NoStateTopic

instance DataSource () Knowledge where
    fetch _ _ _ reqs = SyncFetch $ do
        forM_ reqs $ \(BlockedFetch req var) -> runKnowledge req var
instance DataSource () Topic where
    fetch _ _ _ reqs = SyncFetch $ do
        forM_ reqs $ \(BlockedFetch req var) -> runTopic req var


initialState :: StateStore
initialState = stateSet NoStateKnowledge
             $ stateSet NoStateTopic
             $ stateEmpty


main :: IO ()
main = do
    myEnv <- initEnv initialState ()
    r <- runHaxl myEnv $ do
        f1 <- dataFetch (Topic1 "history")
        dataFetch (Knowledge f1)
    print r
    r' <- runHaxl myEnv $ do
        f2 <- dataFetch (Topic2 "biology")
        dataFetch (Knowledge f2)
    print r'


