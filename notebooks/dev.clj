(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(clay/make! {:format [:quarto :html]
             :base-source-path "notebooks"
             :source-path ["index.clj"
                           "setup.clj"
                           "llamppl/utils.clj"
                           "llamppl/llm.clj"
                           "llamppl/cache.clj"
                           "llamppl/smc.clj"]
             :base-target-path "docs"
             :book {:title "LLaMPPL.clj"}
             :clean-up-target-dir true})
